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
    Encoder_IPM_up,
    MLP
)

def cal_IPM_loss(vae_output, ground_truth, pose_vae):

    loss = (vae_output[:,:,3:6] - ground_truth[:,:,3:6]).pow(2).mean(dim=(0, -1))
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
        ep_ipm_loss = data["ep_ipm_loss"]
        ep_bone_loss = data["ep_bone_loss"]

        print(
            "{} | Recon: {:.3e} | IPM: {:.3e} | Bone: {:.3e}".format(
                self.time_since(ep), ep_recon_loss, ep_ipm_loss, ep_bone_loss
            ),
            flush=True,
        )
def feed_vae_eval(pose_vae, ground_truth, condition, future_weights, latent_z):
    condition = condition.flatten(start_dim=1, end_dim=2)

    output_shape = (-1, pose_vae.num_future_predictions, pose_vae.frame_size_rec)


    vae_output = pose_vae.sample(latent_z, condition)
    vae_output = vae_output.view(output_shape)

    recon_loss = (vae_output - ground_truth).pow(2).mean(dim=(0, -1))
    recon_loss = recon_loss.mul(future_weights).sum()

    return vae_output, recon_loss

def feed_vae(pose_vae, ground_truth, condition, future_weights, latent_z):
    condition = condition.flatten(start_dim=1, end_dim=2)
    #flattened_truth = ground_truth.flatten(start_dim=1, end_dim=2)
    output_shape = (-1, pose_vae.num_future_predictions, pose_vae.frame_size_rec)

    vae_output = pose_vae.sample(latent_z, condition)
    vae_output = vae_output.view(output_shape)

    recon_loss = (vae_output - ground_truth).pow(2).mean(dim=(0, -1))
    recon_loss = recon_loss.mul(future_weights).sum()

    return vae_output, recon_loss


def main():
    env_path = os.path.join(parent_dir, "environments")
    print('cuda:1', 15)
    # setup parameters
    args = SimpleNamespace(
        device="cuda:1" if torch.cuda.is_available() else "cpu",
        mocap_file=os.path.join(env_path, "mocap.npz"),
        norm_mode="zscore",
        latent_size=64,
        num_embeddings=12,
        num_experts=4,
        num_condition_frames=1,
        num_future_predictions=1,
        num_steps_per_rollout=32,
        kl_beta=1.0,
        load_saved_model=True,
    )
    neighbors_list = [1, 2, 3, 4, 6, 10, 5, 7, 8, 9, 11, 12, 13]
    joints_list = [0, 1, 2, 3, 3, 3, 4, 6, 7, 8, 10, 11, 12]
    # learning parameters
    # teacher_epochs = 20
    # ramping_epochs = 20
    # student_epochs = 100
    # args.num_epochs = teacher_epochs + ramping_epochs + student_epochs
    args.num_epochs = 100
    args.mini_batch_size = 64
    args.initial_lr = 1e-4
    args.final_lr = 1e-7

    with open('processed_data/train_data_up_guide_dyn.pkl', 'rb') as f:
        raw_data = pickle.load(f)
    mocap_data = torch.from_numpy(raw_data["data"]).float().to(args.device)
    end_indices = raw_data["end_indices"]

    avg = mocap_data.mean(dim=0)
    std = mocap_data.std(dim=0)
    # Make sure we don't divide by 0
    std[std == 0] = 1.0

    if args.norm_mode == "zscore":
        mocap_data_n = (mocap_data - avg) / std

    batch_size = mocap_data_n.size()[0]
    frame_size = mocap_data_n.size()[1]
    # positions = mocap_data[:, 3:45].reshape(-1, 14, 3)
    # bones_len = torch.norm(positions[:, joints_list, :] - positions[:, neighbors_list, :], dim=-1)

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

    pose_vae_path = 'saved_models_sm/cvaeUp.pt'
    label = pose_vae_path[-5:-3]
    if args.load_saved_model:
        pose_vae = torch.load(pose_vae_path, map_location=args.device)
    pose_vae.eval()
    pose_vae.data_avg = avg[15:]
    pose_vae.data_std = std[15:]

    guide_network = Encoder_IPM_up(ipm_size=15, frame_size=198, latent_size=64, hidden_size=256, label=label).to(args.device)
    guide_network.train()

    guide_optimizer = optim.Adam(guide_network.parameters(), lr=args.initial_lr)
    # guide_optimizer = optim.Adam([{'params': guide_network.parameters()}, {'params': pose_vae.parameters()}],
    #                              lr=args.initial_lr)

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
    sample_schedule = torch.ones(100)

    future_weights = (
        torch.ones(args.num_future_predictions)
        .to(args.device)
        .div_(args.num_future_predictions)
    )

    # buffer for later
    shape = (args.mini_batch_size, args.num_condition_frames, frame_size-15)
    history = torch.empty(shape).to(args.device)

    log_path = os.path.join(current_dir, "log_posevae_progress")
    logger = StatsLogger(args, csv_path=log_path)

    with open('processed_data/test_data_up_guide_dyn.pkl', 'rb') as f:
        test_raw_data = pickle.load(f)
    test_mocap_data_ori = torch.from_numpy(test_raw_data["data"]).float().to(args.device)
    test_mocap_data = torch.from_numpy(test_raw_data["data"]).float().to(args.device)
    test_end_indices = test_raw_data["end_indices"]

    if args.norm_mode == "zscore":
        test_mocap_data_n = (test_mocap_data - avg) / std

    test_batch_size = test_mocap_data.size()[0]
    test_frame_size = test_mocap_data.size()[1]

    test_num_peds = len(test_end_indices)
    test_shape = (1, args.num_condition_frames, test_frame_size - 15)
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
        ep_ipm_loss = 0
        ep_bone_loss = 0
        ep_z_mu_loss = 0
        ep_z_var_loss = 0

        # update_linear_schedule(
        #     vae_optimizer, ep - 1, args.num_epochs, args.initial_lr, args.final_lr
        # )

        for num_mini_batch, indices in enumerate(sampler):
            t_indices = torch.LongTensor(indices)

            # condition is from newest...oldest, i.e. (t-1, t-2, ... t-n)
            condition_range = (
                t_indices.repeat((args.num_condition_frames, 1)).t()
                + torch.arange(args.num_condition_frames - 1, -1, -1).long()
            )

            t_indices += args.num_condition_frames
            history[:, : args.num_condition_frames].copy_(mocap_data_n[condition_range][:,:,15:])
            history_input_guide = mocap_data_n[condition_range].squeeze()

            for offset in range(args.num_steps_per_rollout):
                # dims: (num_parallel, num_window, feature_size)
                use_student = torch.rand(1) < sample_schedule[ep - 1]

                prediction_range = (
                    t_indices.repeat((args.num_future_predictions, 1)).t()
                    + torch.arange(offset, offset + args.num_future_predictions).long()
                )
                ground_truth_con = mocap_data_n[prediction_range][:,:,15:]
                ground_truth_rec = ground_truth_con[:,:,:171]
                condition = history[:, : args.num_condition_frames]
                # ground_truth_bones_len = bones_len[prediction_range].squeeze()

                gt_z, gt_z_mu, gt_z_var = pose_vae.encode(ground_truth_rec[:, 0, :], condition[:, 0, :])
                latent_z, latent_z_mu, latent_z_var = guide_network(history_input_guide[:, :15],
                                                                    history_input_guide[:, 15:])


                # PoseVAE, PoseMixtureVAE, PoseMixtureSpecialistVAE
                vae_output, recon_loss = feed_vae(pose_vae, ground_truth_rec, condition, future_weights, latent_z)
                #IPM_loss = cal_IPM_loss(vae_output, ground_truth_rec, pose_vae)
                #vae_output_dn = pose_vae.denormalize(vae_output.squeeze(), True)
                #generate_positions = vae_output_dn[:,3:45].reshape(-1, 14, 3)
                #generate_bones_len = torch.norm(generate_positions[:, joints_list, :] - generate_positions[:, neighbors_list, :], dim=-1)
                #bone_loss = (generate_bones_len - ground_truth_bones_len).abs().mean()
                z_mu_loss = (latent_z_mu - gt_z_mu).pow(2).mean()
                z_var_loss = (latent_z_var - gt_z_var).pow(2).mean()
                guide_optimizer.zero_grad()
                #total_loss = recon_loss + 10 * IPM_loss
                total_loss = z_mu_loss + z_var_loss
                total_loss.backward()
                guide_optimizer.step()

                history = history.roll(1, dims=1)
                if use_student:
                    next_frame_up = vae_output[:, 0]
                    next_frame = torch.cat((next_frame_up, ground_truth_con.squeeze()[:, 171:]), dim=-1)
                else:
                    next_frame = ground_truth_con[:, 0]
                next_input_guide = torch.cat((mocap_data_n[prediction_range].squeeze()[:, :15], next_frame), dim=-1)
                history[:, 0].copy_(next_frame.detach())
                history_input_guide.copy_(next_input_guide.detach())

                ep_recon_loss += float(recon_loss) / args.num_steps_per_rollout
                ep_z_mu_loss += float(z_mu_loss) / args.num_steps_per_rollout
                ep_z_var_loss += float(z_var_loss) / args.num_steps_per_rollout
                #ep_ipm_loss += float(10 * IPM_loss) / args.num_steps_per_rollout
                #ep_bone_loss += float(bone_loss*0) / args.num_steps_per_rollout

        avg_ep_recon_loss = ep_recon_loss / num_mini_batch
        avg_ep_ipm_loss = ep_ipm_loss / num_mini_batch
        avg_ep_bone_loss = ep_bone_loss / num_mini_batch
        avg_ep_z_mu_loss = ep_z_mu_loss / num_mini_batch
        avg_ep_z_var_loss = ep_z_var_loss / num_mini_batch

        logger.log_stats(
            {
                "epoch": ep,
                "ep_recon_loss": avg_ep_recon_loss,
                "ep_ipm_loss": avg_ep_z_mu_loss,
                "ep_bone_loss": avg_ep_z_var_loss
            }
        )
        #torch.save(copy.deepcopy(guide_network).cpu(), 'saved_models_v1/train_ipm_guide_up_2.pt')

        #test phase
        guide_network.eval()
        pose_vae.eval()
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
            test_history[:, : args.num_condition_frames].copy_(test_mocap_data_n[test_condition_range][:,:,15:])
            test_history_input_guide.copy_(test_mocap_data_n[test_condition_range].squeeze())
            prediction = torch.zeros((test_num_steps, 198)).to(args.device)
            prediction[0] = test_history.squeeze()

            for step in range(1, test_num_steps):
                test_prediction_range = (test_condition_range
                                    + torch.arange(step, step + 1).long()
                                    )
                test_ground_truth_con = test_mocap_data_n[test_prediction_range][:, :, 15:]
                test_ground_truth_rec = test_ground_truth_con[:,:,:171]
                test_condition = test_history[:, : args.num_condition_frames]

                test_gt_z, test_gt_z_mu, test_gt_z_var = pose_vae.encode(test_ground_truth_rec[:, 0], test_history[:, 0])
                test_latent_z, test_latent_z_mu, test_latent_z_var = guide_network(test_history_input_guide[:, :15],
                                                                                   test_history_input_guide[:, 15:])

                test_vae_output, test_recon_loss = feed_vae_eval(pose_vae, test_ground_truth_rec, test_condition, 1,
                                                                 test_latent_z)

                test_z_mu_loss = (test_latent_z_mu - test_gt_z_mu).pow(2).mean()
                test_z_var_loss = (test_latent_z_var - test_gt_z_var).pow(2).mean()
                test_total_loss = test_z_mu_loss + test_z_var_loss
                test_history = test_history.roll(1, dims=1)
                test_next_frame_up = test_vae_output[:, 0]
                test_next_frame = torch.cat((test_next_frame_up, test_ground_truth_con[:, 0, 171:],), dim=-1)
                test_next_input_guide = torch.cat((test_mocap_data_n[test_prediction_range][:, 0, :15], test_next_frame),
                                                  dim=-1)
                test_history[:, 0].copy_(test_next_frame.detach())
                test_history_input_guide.copy_(test_next_input_guide.detach())
                prediction[step] = test_next_frame

                test_ep_recon_loss += float(test_recon_loss)
            prediction_dn = pose_vae.denormalize(prediction)
            prediction_list.append(prediction)
            prediction_dn_list.append(prediction_dn)
        test_ep_recon_loss /= test_count_steps
        prediction_all = torch.cat(prediction_dn_list, dim=0)
        loss_2_global_positions_up = torch.norm(
            prediction_all[:, 3:45].reshape(-1, 14, 3) - test_mocap_data_ori[:, 18:60].reshape(-1, 14, 3), dim=-1).mean()
        if loss_2_global_positions_up < best_test_loss:
            print(
                f'epoch:{ep} per epoch average testing loss improves from {best_test_loss} to {loss_2_global_positions_up}')
            print('current rec_loss:', test_ep_recon_loss)
            best_test_loss = loss_2_global_positions_up
            torch.save(copy.deepcopy(guide_network).cpu(), 'saved_models/guideUp.pt')
            #torch.save(copy.deepcopy(pose_vae).cpu(), 'saved_models/test_cvae_guide_up_nnbss_15d_1.pt')
        guide_network.train()
        pose_vae.train()

if __name__ == "__main__":
    main()