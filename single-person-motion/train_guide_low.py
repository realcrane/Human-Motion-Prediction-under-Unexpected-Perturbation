import copy
import os
import time
from types import SimpleNamespace

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import pickle

from common.logging_utils import CSVLogger
from common.misc_utils import update_linear_schedule
from models import (
    PoseMixtureVAE,
    MLP
)
def frame2ipm(data_dn):
    if len(data_dn.shape) == 1:
        data_dn = data_dn.unsqueeze(0)
    data_positions = data_dn[:,3:30].reshape(-1,9,3)
    data_hip = data_positions[:, 0, :] / 1000
    data_cart = (data_positions[: , 3, :] / 1000 + data_positions[: , 7, :] / 1000) / 2
    rod_direction_norm = torch.norm((data_hip - data_cart), dim=-1).unsqueeze(1)
    rod_direction = (data_hip - data_cart) / rod_direction_norm
    phi = torch.arcsin(-rod_direction[:, 1:2])
    theta = torch.arcsin(rod_direction[:, 0:1] / torch.cos(phi))
    return torch.cat((data_cart[:,:2], theta, phi), dim=1)



def cal_IPM_loss(vae_output, ground_truth, pose_vae):
    vae_output_dn = pose_vae.denormalize(vae_output.squeeze())
    ground_truth_dn = pose_vae.denormalize(ground_truth.squeeze())
    generate_ipm_states = frame2ipm(vae_output_dn)
    ground_truth_ipm_states = frame2ipm(ground_truth_dn)
    loss = (generate_ipm_states - ground_truth_ipm_states).pow(2).mean(dim=(0, -1))
    return loss

class StatsLogger:
    def __init__(self, args, csv_path):
        self.start = time.time()
        self.logger = CSVLogger(log_path=csv_path)
        self.num_epochs = args.num_epochs
        self.progress_format = None

    def time_since(self, ep):
        now = time.time()
        elapsed = now - self.start
        estimated = elapsed * self.num_epochs / ep
        remaining = estimated - elapsed

        em, es = divmod(elapsed, 60)
        rm, rs = divmod(remaining, 60)

        if self.progress_format is None:
            time_format = "%{:d}dm %02ds".format(int(np.log10(rm) + 1))
            perc_format = "%{:d}d %5.1f%%".format(int(np.log10(self.num_epochs) + 1))
            self.progress_format = f"{time_format} (- {time_format}) ({perc_format})"

        return self.progress_format % (em, es, rm, rs, ep, ep / self.num_epochs * 100)

    def log_stats(self, data):
        self.logger.log_epoch(data)

        ep = data["epoch"]
        ep_recon_loss = data["ep_recon_loss"]
        ep_kl_loss = data["ep_kl_loss"]
        ep_ipm_loss = data["ep_ipm_loss"]

        print(
            "{} | Recon: {:.3e} | KL: {:.3e} | IPM: {:.3e}".format(
                self.time_since(ep), ep_recon_loss, ep_kl_loss, ep_ipm_loss
            ),
            flush=True,
        )
def feed_vae_eval(pose_vae, ground_truth, condition, future_weights, latent_z):
    condition = condition.flatten(start_dim=1, end_dim=2)

    output_shape = (-1, pose_vae.num_future_predictions, pose_vae.frame_size)


    vae_output = pose_vae.sample(latent_z, condition)
    vae_output = vae_output.view(output_shape)

    recon_loss = (vae_output - ground_truth).pow(2).mean(dim=(0, -1))
    recon_loss = recon_loss.mul(future_weights).sum()

    return vae_output, recon_loss

def feed_vae(pose_vae, ground_truth, condition, future_weights, latent_z):
    condition = condition.flatten(start_dim=1, end_dim=2)
    #flattened_truth = ground_truth.flatten(start_dim=1, end_dim=2)
    output_shape = (-1, pose_vae.num_future_predictions, pose_vae.frame_size)

    vae_output = pose_vae.sample(latent_z, condition)
    vae_output = vae_output.view(output_shape)

    recon_loss = (vae_output - ground_truth).pow(2).mean(dim=(0, -1))
    recon_loss = recon_loss.mul(future_weights).sum()

    return vae_output, recon_loss


def main():
    env_path = os.path.join(parent_dir, "environments")

    # setup parameters
    args = SimpleNamespace(
        device="cuda:1" if torch.cuda.is_available() else "cpu",
        mocap_file=os.path.join(env_path, "mocap.npz"),
        norm_mode="zscore",
        latent_size=64,
        num_embeddings=12,
        num_experts=6,
        num_condition_frames=1,
        num_future_predictions=1,
        num_steps_per_rollout=32,
        kl_beta=0.1,
        load_saved_model=True,
    )

    # learning parameters
    # teacher_epochs = 20
    # ramping_epochs = 20
    # student_epochs = 100
    # args.num_epochs = teacher_epochs + ramping_epochs + student_epochs
    args.num_epochs = 100
    args.mini_batch_size = 64
    args.initial_lr = 1e-5
    args.final_lr = 1e-7

    with open('processed_data/train_data_low_guide.pkl', 'rb') as f:
        raw_data = pickle.load(f)
    mocap_data = torch.from_numpy(raw_data["data"]).float().to(args.device)
    end_indices = raw_data["end_indices"]

    max = mocap_data.max(dim=0)[0]
    min = mocap_data.min(dim=0)[0]
    avg = mocap_data.mean(dim=0)
    std = mocap_data.std(dim=0)

    # Make sure we don't divide by 0
    std[std == 0] = 1.0

    normalization = {
        "mode": args.norm_mode,
        "max": max,
        "min": min,
        "avg": avg,
        "std": std,
    }
    if args.norm_mode == "minmax":
        mocap_data = 2 * (mocap_data - min) / (max - min) - 1

    elif args.norm_mode == "zscore":
        mocap_data_n = (mocap_data - avg) / std

    batch_size = mocap_data_n.size()[0]
    frame_size = mocap_data_n.size()[1]

    # bad indices are ones that has no required next frames
    # need to take account of num_steps_per_rollout and num_future_predictions
    bad_indices = np.sort(
        np.concatenate(
            [
                end_indices - i
                for i in range(
                    args.num_steps_per_rollout
                    + (args.num_condition_frames - 1)
                    + (args.num_future_predictions - 1)
                )
            ]
        )
    )
    all_indices = np.arange(batch_size)
    good_masks = np.isin(all_indices, bad_indices, assume_unique=True, invert=True)
    selectable_indices = all_indices[good_masks]

    pose_vae = PoseMixtureVAE(
        frame_size,
        args.latent_size,
        args.num_condition_frames,
        args.num_future_predictions,
        normalization,
        args.num_experts,
    ).to(args.device)
    pose_vae_path = 'saved_models/cvaeLow.pt'
    if args.load_saved_model:
        pose_vae = torch.load(pose_vae_path, map_location=args.device)
    pose_vae.eval()
    pose_vae.data_avg = avg[6:]
    pose_vae.data_std = std[6:]

    guide_network = MLP(117, args.latent_size).to(args.device)
    guide_network.train()

    guide_optimizer = optim.Adam(guide_network.parameters(), lr=args.initial_lr)

    # sample_schedule = torch.cat(
    #     (
    #         # First part is pure teacher forcing
    #         torch.zeros(teacher_epochs),
    #         # Second part with schedule sampling
    #         torch.linspace(0.0, 1.0, ramping_epochs),
    #         # last part is pure student
    #         torch.ones(student_epochs),
    #     )
    # )
    sample_schedule = torch.ones(200)
    # future_weights = torch.softmax(
    #     torch.linspace(1, 0, args.num_future_predictions), dim=0
    # ).to(args.device)

    future_weights = (
        torch.ones(args.num_future_predictions)
        .to(args.device)
        .div_(args.num_future_predictions)
    )

    # buffer for later
    shape = (args.mini_batch_size, args.num_condition_frames, frame_size-6)
    history = torch.empty(shape).to(args.device)
    #history_input_guide = torch.empty(shape_input_guide).to(args.device)

    log_path = os.path.join(current_dir, "log_posevae_progress")
    logger = StatsLogger(args, csv_path=log_path)

    with open('processed_data/test_data_low_guide.pkl', 'rb') as f:
        test_raw_data = pickle.load(f)
    test_mocap_data_ori = torch.from_numpy(test_raw_data["data"]).float().to(args.device)
    test_end_indices = test_raw_data["end_indices"]



    if args.norm_mode == "zscore":
        test_mocap_data = (test_mocap_data_ori - avg) / std

    test_batch_size = test_mocap_data.size()[0]
    test_frame_size = test_mocap_data.size()[1]

    test_num_peds = len(test_end_indices)
    test_shape = (1, args.num_condition_frames, test_frame_size-6)
    test_shape_input_guide = (1, frame_size)
    test_history = torch.empty(test_shape).to(args.device)
    test_history_input_guide = torch.empty(test_shape_input_guide).to(args.device)
    best_test_loss = np.inf

    for ep in range(1, args.num_epochs + 1):
        sampler = BatchSampler(
            SubsetRandomSampler(selectable_indices),
            args.mini_batch_size,
            drop_last=True,
        )
        ep_recon_loss = 0
        ep_kl_loss = 0
        ep_perplexity = 0

        # update_linear_schedule(
        #     guide_optimizer, ep - 1, args.num_epochs, args.initial_lr, args.final_lr
        # )

        num_mini_batch = 1
        for num_mini_batch, indices in enumerate(sampler):
            t_indices = torch.LongTensor(indices)

            # condition is from newest...oldest, i.e. (t-1, t-2, ... t-n)
            condition_range = (
                t_indices.repeat((args.num_condition_frames, 1)).t()
                + torch.arange(args.num_condition_frames - 1, -1, -1).long()
            )

            t_indices += args.num_condition_frames
            history[:, : args.num_condition_frames].copy_(mocap_data_n[condition_range][:,:,6:])
            history_input_guide = mocap_data_n[condition_range].squeeze()

            for offset in range(args.num_steps_per_rollout):
                # dims: (num_parallel, num_window, feature_size)
                use_student = torch.rand(1) < sample_schedule[ep - 1]

                prediction_range = (
                    t_indices.repeat((args.num_future_predictions, 1)).t()
                    + torch.arange(offset, offset + args.num_future_predictions).long()
                )
                ground_truth = mocap_data_n[prediction_range][:,:,6:]
                condition = history[:, : args.num_condition_frames]

                latent_z = guide_network(history_input_guide)

                # PoseVAE, PoseMixtureVAE, PoseMixtureSpecialistVAE
                vae_output, recon_loss = feed_vae(pose_vae, ground_truth, condition, future_weights, latent_z)
                IPM_loss = cal_IPM_loss(vae_output, ground_truth, pose_vae)
                guide_optimizer.zero_grad()
                total_loss = recon_loss + 10000 * IPM_loss
                total_loss.backward()
                guide_optimizer.step()

                history = history.roll(1, dims=1)
                next_frame = vae_output[:, 0] if use_student else ground_truth[:, 0]
                next_input_guide = torch.cat((mocap_data_n[prediction_range].squeeze()[:,:6], next_frame), dim=-1)
                history[:, 0].copy_(next_frame.detach())
                history_input_guide.copy_(next_input_guide.detach())


                ep_recon_loss += float(recon_loss) / args.num_steps_per_rollout

        avg_ep_recon_loss = ep_recon_loss / num_mini_batch

        logger.log_stats(
            {
                "epoch": ep,
                "ep_recon_loss": avg_ep_recon_loss,
                "ep_kl_loss": 0,
                "ep_ipm_loss": 10000*IPM_loss,
            }
        )

        #torch.save(copy.deepcopy(guide_network).cpu(), 'saved_models/train_guide_pos.pt')

        #test phase
        guide_network.eval()
        test_ep_recon_loss = 0
        test_count_steps = 0
        prediction_list = []
        prediction_dn_list = []
        for i in range(test_num_peds):
            if i == 0:
                test_num_steps = test_end_indices[0] + 1
                test_condition_range = torch.tensor([[0]])
            else:
                test_num_steps = test_end_indices[i] - test_end_indices[i - 1]
                test_condition_range = torch.tensor(test_end_indices[i - 1]).reshape(1,1).long() + 1
            test_count_steps += test_num_steps - 1
            test_history[:, : args.num_condition_frames].copy_(test_mocap_data[test_condition_range][:,:,6:])
            test_history_input_guide.copy_(test_mocap_data[test_condition_range].squeeze())
            prediction = torch.zeros((test_num_steps, 111)).to(args.device)
            prediction[0] = copy.deepcopy(test_history.squeeze())
            for step in range(1, test_num_steps):
                test_prediction_range = (test_condition_range
                                    + torch.arange(step, step + 1).long()
                                    )
                test_ground_truth = test_mocap_data[test_prediction_range][:,:,6:]
                test_condition = test_history[:, : args.num_condition_frames]
                test_input_guide = test_history_input_guide
                test_latent_z = guide_network(test_input_guide)

                test_vae_output, test_recon_loss = feed_vae_eval(pose_vae, test_ground_truth, test_condition, 1, test_latent_z)

                test_history = test_history.roll(1, dims=1)
                test_next_frame = test_vae_output[:, 0]
                test_next_input_guide = torch.cat((test_mocap_data[test_prediction_range][:,0,:6], test_next_frame), dim=-1)
                test_history[:, 0].copy_(test_next_frame.detach())
                test_history_input_guide.copy_(test_next_input_guide.detach())
                prediction[step] = test_next_frame

                test_ep_recon_loss += float(test_recon_loss)
            prediction_dn = pose_vae.denormalize(prediction)
            prediction_list.append(prediction)
            prediction_dn_list.append(prediction_dn)
        test_ep_recon_loss /= test_count_steps
        prediction_all = torch.cat(prediction_dn_list, dim=0)
        loss_2_global_positions = torch.norm(
            prediction_all[:, 3:30].reshape(-1, 9, 3) - test_mocap_data_ori[:, 9:36].reshape(-1, 9, 3), dim=-1).mean()
        if loss_2_global_positions < best_test_loss:
            print(f'epoch:{ep} per epoch average testing loss improves from {best_test_loss} to {loss_2_global_positions}')
            print('current_rec_loss:', test_ep_recon_loss)
            best_test_loss = loss_2_global_positions
            torch.save(copy.deepcopy(guide_network).cpu(), 'saved_models/guideLow.pt')
        guide_network.train()

if __name__ == "__main__":
    main()