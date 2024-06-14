import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.controller import init, DiagGaussian
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=[256, 256], activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                # if self.dropout != -1:
                #     x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, frame_size, latent_size, normalization):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size

        self.mode = normalization.get("mode")
        self.data_max = normalization.get("max")
        self.data_min = normalization.get("min")
        self.data_avg = normalization.get("avg")
        self.data_std = normalization.get("std")

        h1 = 256
        h2 = 128
        # Encoder
        # Takes pose | condition (n * poses) as input
        self.fc1 = nn.Linear(frame_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, latent_size)

        # Decoder
        # Takes latent | condition as input
        self.fc4 = nn.Linear(latent_size, h2)
        self.fc5 = nn.Linear(h2, h1)
        self.fc6 = nn.Linear(h1, frame_size)

    def normalize(self, t):
        if self.mode == "minmax":
            return 2 * (t - self.data_min) / (self.data_max - self.data_min) - 1
        elif self.mode == "zscore":
            return (t - self.data_avg) / self.data_std
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def denormalize(self, t):
        if self.mode == "minmax":
            return (t + 1) * (self.data_max - self.data_min) / 2 + self.data_min
        elif self.mode == "zscore":
            return t * self.data_std + self.data_avg
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)

    def decode(self, x):
        h4 = F.relu(self.fc4(x))
        h5 = F.relu(self.fc5(h4))
        return self.fc6(h5)

class Encoder_IPM(nn.Module):
    def __init__(
        self,
        ipm_size,
        frame_size,
        latent_size,
        hidden_size,
    ):
        super().__init__()
        # Encoder
        # Takes pose | condition (n * poses) as input
        input_size = ipm_size + frame_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(ipm_size + hidden_size, hidden_size)
        self.fc3 = nn.Linear(ipm_size + hidden_size, hidden_size)
        self.fc4 = nn.Linear(ipm_size + hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.mu = nn.Linear(ipm_size + hidden_size, latent_size)
        self.logvar = nn.Linear(ipm_size + hidden_size, latent_size)
        self.mu_min_04, self.mu_min_13  = -6, -13
        self.mu_max_04, self.mu_max_13 = 7, 10
        self.logvar_min_04, self.logvar_min_13  = -11, -8
        self.logvar_max_04, self.logvar_max_13 = 0, 1
        self.sigmoid = nn.Sigmoid()
        # self.mu_scale_len = scale[:, 0].reshape((1, 64))
        # self.mu_scale_min = scale[:, 1].reshape((1, 64))
        # torch.nn.init.kaiming_uniform_(self.fc1.weight.data)
        # torch.nn.init.kaiming_uniform_(self.fc2.weight.data)
        # torch.nn.init.kaiming_uniform_(self.mu.weight.data)
        # torch.nn.init.kaiming_uniform_(self.logvar.weight.data)

    def encode(self, x, c):
        label = '13'
        h1 = F.elu(self.fc1(torch.cat((x, c), dim=1)))
        #h1 = self.dropout1(h1)
        h2 = F.elu(self.fc2(torch.cat((x, h1), dim=1)))
        #h2 = self.dropout2(h2)
        # h3 = F.elu(self.fc3(torch.cat((x, h2), dim=1)))
        # h4 = F.elu(self.fc4(torch.cat((x, h3), dim=1)))
        s = torch.cat((x, h2), dim=1)
        if label == '04':
            mu = self.sigmoid(self.mu(s)) * 13 - 6
            logvar = self.sigmoid(self.logvar(s)) * 11 - 11
        elif label == '13':
            #mu = self.sigmoid(self.mu(s)) * self.mu_scale_len + self.mu_scale_min
            mu = self.sigmoid(self.mu(s)) * 8 - 4
            logvar = self.sigmoid(self.logvar(s)) * 6.6 - 6.4
        else:
            mu = self.sigmoid(self.mu(s)) * 15 - 7
            logvar = self.sigmoid(self.logvar(s)) * 5.1 - 5
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class Encoder_IPM_full(nn.Module):
    def __init__(
        self,
        ipm_size,
        frame_size,
        latent_size,
        hidden_size
    ):
        super().__init__()
        # Encoder
        # Takes pose | condition (n * poses) as input
        input_size = ipm_size + frame_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(ipm_size + hidden_size, hidden_size)
        self.fc3 = nn.Linear(ipm_size + hidden_size, hidden_size)
        self.fc4 = nn.Linear(ipm_size + hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.mu = nn.Linear(ipm_size + hidden_size, latent_size)
        self.logvar = nn.Linear(ipm_size + hidden_size, latent_size)
        self.mu_min_04, self.mu_min_13  = -6, -13
        self.mu_max_04, self.mu_max_13 = 7, 10
        self.logvar_min_04, self.logvar_min_13  = -11, -8
        self.logvar_max_04, self.logvar_max_13 = 0, 1
        self.sigmoid = nn.Sigmoid()


    def encode(self, x, c):
        label = '13'
        h1 = F.elu(self.fc1(torch.cat((x, c), dim=1)))
        h2 = F.elu(self.fc2(torch.cat((x, h1), dim=1)))
        s = torch.cat((x, h2), dim=1)

        mu = self.sigmoid(self.mu(s)) * 22 - 8
        logvar = self.sigmoid(self.logvar(s)) * 3.8 - 3.3

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class Encoder_IPM_up(nn.Module):
    def __init__(
        self,
        ipm_size,
        frame_size,
        latent_size,
        hidden_size,
            label
    ):
        super().__init__()
        # Encoder
        # Takes pose | condition (n * poses) as input
        input_size = ipm_size + frame_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(ipm_size + hidden_size, hidden_size)
        self.mu = nn.Linear(ipm_size + hidden_size, latent_size)
        self.logvar = nn.Linear(ipm_size + hidden_size, latent_size)
        self.sigmoid = nn.Sigmoid()
        self.label = label

    # def encode(self, x, c):
    #     h1 = F.elu(self.fc1(torch.cat((x, c), dim=1)))
    #     h2 = F.elu(self.fc2(torch.cat((x, h1), dim=1)))
    #     s = torch.cat((x, h2), dim=1)
    #     label = '24'
    #     if self.label == '24':
    #         mu = self.sigmoid(self.mu(s)) * 32 - 12
    #         logvar = self.sigmoid(self.logvar(s)) * 6.2 - 6
    #     elif self.label == '15':
    #         mu = self.sigmoid(self.mu(s)) * 30 - 18
    #         logvar = self.sigmoid(self.logvar(s)) * 10.3 - 10
    #     else:
    #         # mu = self.sigmoid(self.mu(s)) * 15 - 7
    #         # logvar = self.sigmoid(self.logvar(s)) * 5.1 - 5
    #         print('error')
    #     return mu, logvar

    def encode(self, x, c):
        h1 = F.elu(self.fc1(torch.cat((x, c), dim=1)))
        h2 = F.elu(self.fc2(torch.cat((x, h1), dim=1)))
        s = torch.cat((x, h2), dim=1)
        return self.mu(s), self.logvar(s)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class Encoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
    ):
        super().__init__()
        # Encoder
        # Takes pose | condition (n * poses) as input
        input_size = frame_size * (num_future_predictions + num_condition_frames)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(frame_size + hidden_size, hidden_size)
        self.mu = nn.Linear(frame_size + hidden_size, latent_size)
        self.logvar = nn.Linear(frame_size + hidden_size, latent_size)

    def encode(self, x, c):
        h1 = F.elu(self.fc1(torch.cat((x, c), dim=1)))
        h2 = F.elu(self.fc2(torch.cat((x, h1), dim=1)))
        s = torch.cat((x, h2), dim=1)
        return self.mu(s), self.logvar(s)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
    ):
        super().__init__()
        # Decoder
        # Takes latent | condition as input
        input_size = latent_size + frame_size * num_condition_frames
        output_size = num_future_predictions * frame_size
        self.fc4 = nn.Linear(input_size, hidden_size)
        self.fc5 = nn.Linear(latent_size + hidden_size, hidden_size)
        self.out = nn.Linear(latent_size + hidden_size, output_size)

    def decode(self, z, c):
        h4 = F.elu(self.fc4(torch.cat((z, c), dim=1)))
        h5 = F.elu(self.fc5(torch.cat((z, h4), dim=1)))
        return self.out(torch.cat((z, h5), dim=1))

    def forward(self, z, c):
        return self.decode(z, c)


class MixedDecoder(nn.Module):
    def __init__(
        self,
        frame_size,  #267
        latent_size, #32
        hidden_size, #256
        num_condition_frames,   #1
        num_future_predictions, #1
        num_experts, #6
    ):
        super().__init__()

        input_size = latent_size + frame_size * num_condition_frames #170
        inter_size = latent_size + hidden_size #288
        output_size = num_future_predictions * frame_size #126
        self.decoder_layers = [
            (
                nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
                nn.Parameter(torch.empty(num_experts, output_size)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.decoder_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)

        # Gating network
        gate_hsize = 64
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, num_experts),
        )

    def forward(self, z, c):
        coefficients = F.softmax(self.gate(torch.cat((z, c), dim=1)), dim=1) #64*6
        layer_out = c

        for (weight, bias, activation) in self.decoder_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            ) #64*170*288

            input = torch.cat((z, layer_out), dim=1).unsqueeze(1) #64*1*170
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1) #64*1*288
            out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
            layer_out = activation(out) if activation is not None else out

        return layer_out


class PoseMixtureVAE(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        num_condition_frames,
        num_future_predictions,
        normalization,
        num_experts,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_future_predictions = num_future_predictions

        self.mode = normalization.get("mode")
        self.data_max = normalization.get("max")
        self.data_min = normalization.get("min")
        self.data_avg = normalization.get("avg")
        self.data_std = normalization.get("std")

        hidden_size = 256
        args = (
            frame_size,
            latent_size,
            hidden_size,
            num_condition_frames,
            num_future_predictions,
        )

        self.encoder = Encoder(*args)
        self.decoder = MixedDecoder(*args, num_experts)

    def normalize(self, t):
        if self.mode == "minmax":
            return 2 * (t - self.data_min) / (self.data_max - self.data_min) - 1
        elif self.mode == "zscore":
            return (t - self.data_avg) / self.data_std
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def denormalize(self, t, if_np = 0):
        if self.mode == "minmax":
            return (t + 1) * (self.data_max - self.data_min) / 2 + self.data_min
        elif self.mode == "zscore":
            if if_np:
                return t * self.data_std.detach().cpu().numpy() + self.data_avg.detach().cpu().numpy()
            else:
                return t * self.data_std + self.data_avg
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def encode(self, x, c):
        z, mu, logvar = self.encoder(x, c)
        return z, mu, logvar

    def forward(self, x, c):
        z, mu, logvar = self.encoder(x, c)
        return self.decoder(z, c), mu, logvar

    def sample(self, z, c, deterministic=False):
        return self.decoder(z, c)



class PoseMixtureSpecialistVAE(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        num_condition_frames,
        num_future_predictions,
        normalization,
        num_experts,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_future_predictions = num_future_predictions

        self.mode = normalization.get("mode")
        self.data_max = normalization.get("max")
        self.data_min = normalization.get("min")
        self.data_avg = normalization.get("avg")
        self.data_std = normalization.get("std")

        hidden_size = 128
        args = (
            frame_size,
            latent_size,
            hidden_size,
            num_condition_frames,
            num_future_predictions,
        )

        self.encoder = Encoder(*args)

        self.decoders = []
        for i in range(num_experts):
            decoder = Decoder(*args)
            self.decoders.append(decoder)
            self.add_module("d" + str(i), decoder)

        # Gating network
        gate_hsize = 128
        input_size = latent_size + frame_size * num_condition_frames
        self.g_fc1 = nn.Linear(input_size, gate_hsize)
        self.g_fc2 = nn.Linear(latent_size + gate_hsize, gate_hsize)
        self.g_fc3 = nn.Linear(latent_size + gate_hsize, num_experts)

    def normalize(self, t):
        if self.mode == "minmax":
            return 2 * (t - self.data_min) / (self.data_max - self.data_min) - 1
        elif self.mode == "zscore":
            return (t - self.data_avg) / self.data_std
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def denormalize(self, t):
        if self.mode == "minmax":
            return (t + 1) * (self.data_max - self.data_min) / 2 + self.data_min
        elif self.mode == "zscore":
            return t * self.data_std + self.data_avg
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def gate(self, z, c):
        h1 = F.elu(self.g_fc1(torch.cat((z, c), dim=1)))
        h2 = F.elu(self.g_fc2(torch.cat((z, h1), dim=1)))
        return self.g_fc3(torch.cat((z, h2), dim=1))

    def forward(self, x, c):
        z, mu, logvar = self.encoder(x, c)
        coefficients = F.softmax(self.gate(z, c), dim=1)
        predictions = torch.stack([decoder(z, c) for decoder in self.decoders], dim=1)
        return predictions, mu, logvar, coefficients

    def sample(self, z, c, deterministic=False):
        coefficients = F.softmax(self.gate(z, c), dim=1)
        predictions = torch.stack([decoder(z, c) for decoder in self.decoders], dim=1)

        if not deterministic:
            dist = torch.distributions.Categorical(coefficients)
            indices = dist.sample()
        else:
            indices = coefficients.argmax(dim=1)

        return predictions[torch.arange(predictions.size(0)), indices]


class PoseVAE(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        num_condition_frames,
        num_future_predictions,
        normalization,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_future_predictions = num_future_predictions

        self.mode = normalization.get("mode")
        self.data_max = normalization.get("max")
        self.data_min = normalization.get("min")
        self.data_avg = normalization.get("avg")
        self.data_std = normalization.get("std")

        h1 = 256
        # Encoder
        # Takes pose | condition (n * poses) as input
        self.fc1 = nn.Linear(
            frame_size * (num_future_predictions + num_condition_frames), h1
        )
        self.fc2 = nn.Linear(frame_size + h1, h1)
        # self.fc3 = nn.Linear(h1, h1)
        self.mu = nn.Linear(frame_size + h1, latent_size)
        self.logvar = nn.Linear(frame_size + h1, latent_size)

        # Decoder
        # Takes latent | condition as input
        self.fc4 = nn.Linear(latent_size + frame_size * num_condition_frames, h1)
        self.fc5 = nn.Linear(latent_size + h1, h1)
        # self.fc6 = nn.Linear(latent_size + h1, h1)
        self.out = nn.Linear(latent_size + h1, num_future_predictions * frame_size)

    def normalize(self, t):
        if self.mode == "minmax":
            return 2 * (t - self.data_min) / (self.data_max - self.data_min) - 1
        elif self.mode == "zscore":
            return (t - self.data_avg) / self.data_std
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def denormalize(self, t):
        if self.mode == "minmax":
            return (t + 1) * (self.data_max - self.data_min) / 2 + self.data_min
        elif self.mode == "zscore":
            return t * self.data_std + self.data_avg
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def encode(self, x, c):
        h1 = F.elu(self.fc1(torch.cat((x, c), dim=1)))
        h2 = F.elu(self.fc2(torch.cat((x, h1), dim=1)))
        # h3 = F.elu(self.fc3(h2))
        s = torch.cat((x, h2), dim=1)
        return self.mu(s), self.logvar(s)

    def decode(self, z, c):
        h4 = F.elu(self.fc4(torch.cat((z, c), dim=1)))
        h5 = F.elu(self.fc5(torch.cat((z, h4), dim=1)))
        # h6 = F.elu(self.fc6(torch.cat((z, h5), dim=1)))
        return self.out(torch.cat((z, h5), dim=1))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample(self, z, c, deterministic=False):
        return self.decode(z, c)

class Encoder_up(nn.Module):
    def __init__(
        self,
        frame_size_con,
        frame_size_rec,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
    ):
        super().__init__()
        # Encoder
        # Takes pose | condition (n * poses) as input
        input_size = frame_size_con + frame_size_rec
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(frame_size_rec + hidden_size, hidden_size)
        self.mu = nn.Linear(frame_size_rec + hidden_size, latent_size)
        self.logvar = nn.Linear(frame_size_rec + hidden_size, latent_size)

    def encode(self, x, c):
        h1 = F.elu(self.fc1(torch.cat((x, c), dim=1)))
        h2 = F.elu(self.fc2(torch.cat((x, h1), dim=1)))
        s = torch.cat((x, h2), dim=1)
        return self.mu(s), self.logvar(s)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder_up(nn.Module):
    def __init__(
        self,
        frame_size_con,
        frame_size_rec,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
    ):
        super().__init__()
        # Decoder
        # Takes latent | condition as input
        input_size = latent_size + frame_size_con * num_condition_frames
        output_size = num_future_predictions * frame_size_rec
        self.fc4 = nn.Linear(input_size, hidden_size)
        self.fc5 = nn.Linear(latent_size + hidden_size, hidden_size)
        self.out = nn.Linear(latent_size + hidden_size, output_size)

    def decode(self, z, c):
        h4 = F.elu(self.fc4(torch.cat((z, c), dim=1)))
        h5 = F.elu(self.fc5(torch.cat((z, h4), dim=1)))
        return self.out(torch.cat((z, h5), dim=1))

    def forward(self, z, c):
        return self.decode(z, c)

class MixedDecoder_up(nn.Module):
    def __init__(
        self,
        frame_size_con,
        frame_size_rec,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
        num_experts,
    ):
        super().__init__()

        input_size = latent_size + frame_size_con * num_condition_frames
        inter_size = latent_size + hidden_size
        output_size = num_future_predictions * frame_size_rec
        self.decoder_layers = [
            (
                nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
                nn.Parameter(torch.empty(num_experts, output_size)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.decoder_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)

        # Gating network
        gate_hsize = 64
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, num_experts),
        )

    def forward(self, z, c):
        coefficients = F.softmax(self.gate(torch.cat((z, c), dim=1)), dim=1) #64*6
        layer_out = c

        for (weight, bias, activation) in self.decoder_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            ) #64*170*288

            input = torch.cat((z, layer_out), dim=1).unsqueeze(1) #64*1*170
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1) #64*1*288
            out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
            layer_out = activation(out) if activation is not None else out

        return layer_out

class PoseMixtureVAE_up(nn.Module):
    def __init__(
        self,
        frame_size_con,
        frame_size_rec,
        latent_size,
        num_condition_frames,
        num_future_predictions,
        normalization,
        num_experts,
    ):
        super().__init__()
        self.frame_size_con = frame_size_con
        self.frame_size_rec = frame_size_rec
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_future_predictions = num_future_predictions

        self.mode = normalization.get("mode")
        self.data_max = normalization.get("max")
        self.data_min = normalization.get("min")
        self.data_avg = normalization.get("avg")
        self.data_std = normalization.get("std")

        hidden_size = 256
        args = (
            frame_size_con,
            frame_size_rec,
            latent_size,
            hidden_size,
            num_condition_frames,
            num_future_predictions,
        )

        self.encoder = Encoder_up(*args)
        self.decoder = MixedDecoder_up(*args, num_experts)

    def normalize(self, t):
        if self.mode == "minmax":
            return 2 * (t - self.data_min) / (self.data_max - self.data_min) - 1
        elif self.mode == "zscore":
            return (t - self.data_avg) / self.data_std
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def denormalize(self, t, if_rec=False):
        if if_rec:
            data_std = self.data_std[:171]
            data_avg = self.data_avg[:171]
            return t * data_std + data_avg

        else:
            if self.mode == "minmax":
                return (t + 1) * (self.data_max - self.data_min) / 2 + self.data_min
            elif self.mode == "zscore":
                return t * self.data_std + self.data_avg
            elif self.mode == "none":
                return t
            else:
                raise ValueError("Unknown normalization mode")

    def encode(self, x, c):
        z, mu, logvar = self.encoder(x, c)
        return z, mu, logvar

    def forward(self, x, c):
        z, mu, logvar = self.encoder(x, c)
        return self.decoder(z, c), mu, logvar

    def sample(self, z, c, deterministic=False):
        return self.decoder(z, c)