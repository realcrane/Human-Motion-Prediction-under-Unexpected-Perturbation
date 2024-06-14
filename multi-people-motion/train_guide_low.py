import copy
import os
import time
from types import SimpleNamespace
from scipy.spatial.transform import Rotation

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import pickle
from torch.utils.tensorboard import SummaryWriter
import pdb
from torch.nn.utils import clip_grad_norm_

from common.logging_utils import CSVLogger
from common.misc_utils import update_linear_schedule, update_linear_schedule_2
from models import (
    PoseMixtureVAE,
    MLP, Encoder_IPM
)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def nan_or_inf(x):
    count = torch.isnan(x).sum()
    if count > 0:
        return 1
    count = torch.isinf(x).sum()
    if count > 0:
        return 2
    return 0
def from_6D_to_rm(orientation_6D):
    a1, a2 = orientation_6D[..., [0, 2, 4]], orientation_6D[..., [1, 3, 5]]
    b1 = F.normalize(a1, dim=-1)
    # diff = a1 - b1
    # diff_norm = torch.abs(diff).sum()
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1,b2,b3), dim=3)
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
        ep_ipm_loss = data["ep_ipm_loss"]
        ep_foot_loss = data["ep_foot_loss"]
        ep_bone_loss = data['ep_bone_loss']

        print(
            "{} | Recon: {:.4e} | IPM: {:.4e} | foot: {:.4e} | bone: {:.4e}".format(
                self.time_since(ep), ep_recon_loss, ep_ipm_loss, ep_foot_loss, ep_bone_loss
            ),
            flush=True,
        )
def feed_vae_eval(pose_vae, ground_truth, condition, future_weights, latent_z):
    #condition = condition.flatten(start_dim=1, end_dim=2)

    output_shape = (-1, pose_vae.frame_size)


    vae_output = pose_vae.sample(latent_z, condition)
    vae_output = vae_output.view(output_shape)

    recon_loss = (vae_output - ground_truth).pow(2).mean()

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
    print('cuda 1', 100)
    logger_curve = SummaryWriter('runs/2')
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
        load_saved_model=True,
    )
    neighbors_list = [1, 5, 2, 3, 4, 6, 7, 8]
    joints_list = [0, 0, 1, 2, 3, 5, 6, 7]
    # learning parameters
    # teacher_epochs = 20
    # ramping_epochs = 20
    # student_epochs = 100
    # args.num_epochs = teacher_epochs + ramping_epochs + student_epochs
    args.num_epochs = 100
    args.mini_batch_size = 64
    args.initial_lr = 1e-4
    args.final_lr = 1e-7

    with open('processed_data/train_data_low_guide_dyn.pkl', 'rb') as f:
        raw_data = pickle.load(f)
    mocap_data_ori = torch.from_numpy(raw_data["data"]).float().to(args.device)
    foot_skate = torch.from_numpy(raw_data["data"]).float().to(args.device)
    end_indices = raw_data["end_indices"]
    bvh_order = 'zyx'

    mocap_data_ori_pos = mocap_data_ori[:, 18:45].reshape((-1, 9, 3))
    mocap_data_ori_rot = mocap_data_ori[:, 72:].reshape((-1, 9, 6))
    mocap_data_ori_rm = from_6D_to_rm(mocap_data_ori_rot).cpu().detach().numpy()
    euler = np.zeros((len(mocap_data_ori), 9, 3))
    for joint in range(9):
        r = Rotation.from_matrix(mocap_data_ori_rm[:, joint])
        euler[:, joint] = r.as_euler(bvh_order, degrees=True)
    euler = torch.from_numpy(euler)

    max = mocap_data_ori.max(dim=0)[0]
    min = mocap_data_ori.min(dim=0)[0]
    avg = mocap_data_ori.mean(dim=0)
    std = mocap_data_ori.std(dim=0)

    print('mean_std:', std[18:45].mean())

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
        mocap_data = 2 * (mocap_data_ori - min) / (max - min) - 1

    elif args.norm_mode == "zscore":
        mocap_data = (mocap_data_ori - avg) / std

    batch_size = mocap_data.size()[0]
    frame_size = mocap_data.size()[1]

    positions = mocap_data_ori[:, 18:45].reshape(-1,9,3)
    bones_len = torch.norm(positions[:, joints_list,:] - positions[:, neighbors_list, :], dim=-1)

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

    pose_vae_path = 'saved_models/cvaeLow.pt'
    label = pose_vae_path[-5:-3]
    pose_vae = torch.load(pose_vae_path, map_location=args.device)
    pose_vae.eval()
    #pose_vae.train()
    pose_vae.data_avg = avg[15:]
    pose_vae.data_std = std[15:]

    #guide_network = MLP(117, args.latent_size, hidden_size=[512, 512]).to(args.device)
    guide_network = Encoder_IPM(ipm_size=15, frame_size=111, latent_size=64, hidden_size=256).to(args.device)
    guide_network.train()

    #guide_optimizer = optim.SGD(guide_network.parameters(), lr=args.initial_lr, momentum=0.8)
    guide_optimizer = optim.Adam(guide_network.parameters(), lr=args.initial_lr)
    #guide_optimizer = optim.AdamW(guide_network.parameters(), lr=args.initial_lr, weight_decay=0.01)
    #guide_optimizer = optim.Adam([{'params': guide_network.parameters()}, {'params': pose_vae.decoder.parameters(), 'lr': 0.1*args.initial_lr}], lr=args.initial_lr)

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
    # sample_schedule = torch.cat(
    #     (
    #         torch.zeros(teacher_epochs_1),
    #         torch.linspace(0.0, 1.0, ramping_epochs_1),
    #         torch.zeros(teacher_epochs_2),
    #         torch.linspace(0.0, 1.0, ramping_epochs_2),
    #     )
    # )


    future_weights = (
        torch.ones(args.num_future_predictions)
        .to(args.device)
        .div_(args.num_future_predictions)
    )

    # buffer for later
    num_peds = len(end_indices)
    shape = (args.mini_batch_size, args.num_condition_frames, frame_size-15)
    history = torch.empty(shape).to(args.device)
    #history_input_guide = torch.empty(shape_input_guide).to(args.device)

    log_path = os.path.join(current_dir, "log_posevae_progress")
    logger = StatsLogger(args, csv_path=log_path)

    with open('processed_data/test_data_low_guide_dyn.pkl', 'rb') as f:
        test_raw_data = pickle.load(f)
    test_mocap_data_ori = torch.from_numpy(test_raw_data["data"]).float().to(args.device)
    test_foot_skate = torch.from_numpy(test_raw_data["data"]).float().to(args.device)
    test_end_indices = test_raw_data["end_indices"]

    if args.norm_mode == "zscore":
        test_mocap_data = (test_mocap_data_ori - avg) / std

    test_batch_size = test_mocap_data.size()[0]
    test_frame_size = test_mocap_data.size()[1]

    test_mocap_data_ori_pos = test_mocap_data_ori[:, 18:45].reshape((-1, 9, 3))
    test_mocap_data_ori_rot = test_mocap_data_ori[:, 72:].reshape((-1, 9, 6))
    test_mocap_data_ori_rm = from_6D_to_rm(test_mocap_data_ori_rot).cpu().detach().numpy()
    test_euler = np.zeros((len(test_mocap_data_ori), 9, 3))
    for joint in range(9):
        test_r = Rotation.from_matrix(test_mocap_data_ori_rm[:, joint])
        test_euler[:, joint] = test_r.as_euler(bvh_order, degrees=True)
    test_euler = torch.from_numpy(test_euler)

    test_num_peds = len(test_end_indices)
    test_shape = (1, test_frame_size-15)
    test_history = torch.empty(test_shape).to(args.device)
    best_test_loss = np.inf

    # with open('processed_data/test_low_guide_dyn_pred_nnbss_15.pkl', 'rb') as f:
    #     testp_raw_data = pickle.load(f)
    # testp_mocap_data_ori = torch.from_numpy(testp_raw_data["data"]).float().to(args.device)[:,:-2]
    # testp_foot_skate = torch.from_numpy(testp_raw_data["data"]).float().to(args.device)[:, -2:]
    # testp_end_indices = testp_raw_data["end_indices"]
    #
    # if args.norm_mode == "zscore":
    #     testp_mocap_data = (testp_mocap_data_ori - avg) / std
    #
    # testp_batch_size = testp_mocap_data.size()[0]
    # testp_frame_size = testp_mocap_data.size()[1]
    #
    # testp_mocap_data_ori_pos = testp_mocap_data_ori[:, 18:45].reshape((-1, 9, 3))
    # testp_mocap_data_ori_rot = testp_mocap_data_ori[:, 72:].reshape((-1, 9, 6))
    # testp_mocap_data_ori_rm = from_6D_to_rm(testp_mocap_data_ori_rot).cpu().detach().numpy()
    # testp_euler = np.zeros((len(testp_mocap_data_ori), 9, 3))
    # for joint in range(9):
    #     testp_r = Rotation.from_matrix(testp_mocap_data_ori_rm[:, joint])
    #     testp_euler[:, joint] = testp_r.as_euler(bvh_order, degrees=True)
    # testp_euler = torch.from_numpy(testp_euler)
    #
    # testp_num_peds = len(testp_end_indices)
    # testp_shape = (1, testp_frame_size-15)
    # testp_history = torch.empty(testp_shape).to(args.device)
    # best_testp_loss = np.inf

    for ep in range(1, args.num_epochs + 1):
        sampler = BatchSampler(
            SubsetRandomSampler(selectable_indices),
            args.mini_batch_size,
            drop_last=True,
        )
        ep_recon_loss = 0
        ep_ipm_loss = 0
        ep_foot_loss = 0
        ep_bone_loss = 0
        ep_recon_loss_phy_pos = 0
        ep_recon_loss_phy_rot = 0
        ep_z_mu_loss = 0
        ep_z_var_loss = 0
        ep_total_loss = 0

        update_linear_schedule(
            guide_optimizer, ep - 1, args.num_epochs, args.initial_lr, args.final_lr
        )

        num_mini_batch = 1
        for num_mini_batch, indices in enumerate(sampler):
            t_indices = torch.LongTensor(indices)

            # condition is from newest...oldest, i.e. (t-1, t-2, ... t-n)
            condition_range = (
                t_indices.repeat((args.num_condition_frames, 1)).t()
                + torch.arange(args.num_condition_frames - 1, -1, -1).long()
            )

            t_indices += args.num_condition_frames
            history[:, : args.num_condition_frames].copy_(mocap_data[condition_range][:,:,15:])
            history_input_guide = mocap_data[condition_range].squeeze()

            for offset in range(args.num_steps_per_rollout):
                # dims: (num_parallel, num_window, feature_size)
                use_student = torch.rand(1) < sample_schedule[ep - 1]

                prediction_range = (
                    t_indices.repeat((args.num_future_predictions, 1)).t()
                    + torch.arange(offset, offset + args.num_future_predictions).long()
                )
                ground_truth = mocap_data[prediction_range][:,:,15:]
                # ground_truth_bones_len = bones_len[prediction_range].squeeze()
                # ground_truth_foot_labels = foot_skate[prediction_range].squeeze()

                gt_pos = mocap_data_ori_pos[prediction_range].squeeze()
                gt_euler = euler[prediction_range].squeeze()

                condition = history[:, : args.num_condition_frames]
                gt_z, gt_z_mu, gt_z_var = pose_vae.encode(ground_truth[:, 0, :], condition[:, 0, :])
                latent_z, latent_z_mu, latent_z_var = guide_network(history_input_guide[:,:15], history_input_guide[:, 15:])
                # PoseVAE, PoseMixtureVAE, PoseMixtureSpecialistVAE

                vae_output, recon_loss = feed_vae(pose_vae, ground_truth, condition, future_weights, latent_z)
                #IPM_loss = cal_IPM_loss(vae_output, ground_truth, pose_vae)
                #vae_output_dn = pose_vae.denormalize(vae_output.squeeze())
                # generate_positions = vae_output_dn[:, 9:36].reshape(-1, 9, 3)
                # generate_vel_from_0 = generate_positions[1:, [3, 7], :] - generate_positions[:-1, [3, 7], :]
                # generate_vel_from_0_norm = torch.norm(generate_vel_from_0, dim=-1)
                # foot_loss = (generate_vel_from_0_norm * ground_truth_foot_labels[:-1]).mean()
                # generate_bones_len = torch.norm(
                #     generate_positions[:, joints_list, :] - generate_positions[:, neighbors_list, :], dim=-1)
                # bone_loss = (generate_bones_len - ground_truth_bones_len).abs().mean()
                # z_mu_loss = torch.norm(latent_z_mu - gt_z_mu, dim=-1).mean()
                # z_var_loss = torch.norm(latent_z_var - gt_z_var, dim=-1).mean()
                z_mu_loss = (latent_z_mu - gt_z_mu).pow(2).mean()
                z_var_loss = (latent_z_var - gt_z_var).pow(2).mean()

                guide_optimizer.zero_grad()
                #total_loss = z_mu_loss
                total_loss = z_mu_loss + z_var_loss
                #total_loss = recon_loss + 100 * IPM_loss
                #total_loss = recon_loss + 100 * IPM_loss + foot_loss/200000 + bone_loss/100000
                total_loss.backward()
                guide_optimizer.step()


                history = history.roll(1, dims=1)
                next_frame = vae_output[:, 0] if use_student else ground_truth[:, 0]
                next_input_guide = torch.cat((mocap_data[prediction_range].squeeze()[:,:15], next_frame), dim=-1)
                history[:, 0].copy_(next_frame.detach())
                history_input_guide.copy_(next_input_guide.detach())

                ep_total_loss += float(total_loss) / args.num_steps_per_rollout
                ep_recon_loss += float(recon_loss) / args.num_steps_per_rollout
                ep_z_mu_loss += float(z_mu_loss) / args.num_steps_per_rollout
                ep_z_var_loss += float(z_var_loss) / args.num_steps_per_rollout
                # ep_ipm_loss += float(IPM_loss) / args.num_steps_per_rollout
                # ep_foot_loss += float(foot_loss) / args.num_steps_per_rollout
                # ep_bone_loss += float(bone_loss) / args.num_steps_per_rollout

                vae_output_dn = pose_vae.denormalize(vae_output.detach()[:, 0])
                pred_pos = vae_output_dn[:, 3:30].reshape((-1, 9, 3))
                pred_rot = vae_output_dn[:, 57:].reshape((-1, 9, 6))
                pred_rm = from_6D_to_rm(pred_rot).cpu().detach().numpy()
                pred_euler = np.zeros((len(pred_rm), 9, 3))
                for joint in range(9):
                    r = Rotation.from_matrix(pred_rm[:, joint])
                    pred_euler[:, joint] = r.as_euler(bvh_order, degrees=True)
                pred_euler = torch.from_numpy(pred_euler)

                ep_recon_loss_phy_pos += torch.norm(gt_pos - pred_pos, dim=-1).mean() / args.num_steps_per_rollout
                ep_recon_loss_phy_rot += torch.abs(gt_euler - pred_euler).mean() / args.num_steps_per_rollout

        avg_ep_total_loss = ep_total_loss / num_mini_batch
        avg_ep_recon_loss = ep_recon_loss / num_mini_batch
        avg_ep_ipm_loss = ep_ipm_loss / num_mini_batch
        avg_ep_foot_loss = ep_foot_loss / num_mini_batch
        avg_ep_bone_loss = ep_bone_loss/ num_mini_batch
        avg_ep_z_mu_loss = ep_z_mu_loss / num_mini_batch
        avg_ep_z_var_loss = ep_z_var_loss / num_mini_batch

        avg_ep_recon_loss_phy_pos = ep_recon_loss_phy_pos / num_mini_batch
        avg_ep_recon_loss_phy_rot = ep_recon_loss_phy_rot / num_mini_batch
        logger_curve.add_scalar('train/total_loss_1', avg_ep_total_loss, ep)
        logger_curve.add_scalar('train/recon_loss_1', avg_ep_recon_loss, ep)
        logger_curve.add_scalar('train/z_mu_loss_1', avg_ep_z_mu_loss, ep)
        logger_curve.add_scalar('train/z_var_loss_1', avg_ep_z_var_loss, ep)
        logger_curve.add_scalar('train/recon_loss_1_pos', avg_ep_recon_loss_phy_pos / 1000, ep)
        logger_curve.add_scalar('train/recon_loss_1_rot', avg_ep_recon_loss_phy_rot, ep)

        logger.log_stats(
            {
                "epoch": ep,
                "ep_recon_loss": avg_ep_recon_loss,
                "ep_ipm_loss": avg_ep_z_mu_loss,
                "ep_foot_loss": avg_ep_z_var_loss,
                "ep_bone_loss": 0
            }
        )

        guide_network.eval()
        with torch.no_grad():
            prediction_dn_list = []
            ep_total_loss_2 = 0
            ep_recon_loss_2 = 0
            ep_z_mu_loss_2 = 0
            ep_z_var_loss_2 = 0
            count_steps = 0
            for i in range(num_peds):
                if i == 0:
                    num_steps = end_indices[0] + 1
                    condition_range = torch.tensor([[0]])
                else:
                    num_steps = end_indices[i] - end_indices[i - 1]
                    condition_range = torch.tensor(end_indices[i - 1]).reshape(1, 1).long() + 1
                count_steps += num_steps - 1
                test_history.copy_(mocap_data[condition_range][:, 0, 15:])
                history_input_guide = mocap_data[condition_range].reshape((1,-1))
                prediction = np.zeros((num_steps, 111))
                prediction[0] = copy.deepcopy(test_history.squeeze().detach().cpu().numpy())
                for step in range(1, num_steps):
                    prediction_range = (condition_range
                                             + torch.arange(step, step + 1).long()
                                             )
                    ground_truth = mocap_data[prediction_range][:, 0, 15:]
                    gt_z, gt_z_mu, gt_z_var = pose_vae.encode(ground_truth, test_history)
                    latent_z, latent_z_mu, latent_z_var = guide_network(history_input_guide[:, :15], history_input_guide[:, 15:])

                    next_frame_vae_output, recon_loss = feed_vae_eval(pose_vae, ground_truth, test_history, 1,
                                                                     latent_z)
                    z_mu_loss = (latent_z_mu - gt_z_mu).pow(2).mean()
                    z_var_loss = (latent_z_var - gt_z_var).pow(2).mean()
                    total_loss = z_mu_loss + z_var_loss

                    next_input_guide = torch.cat((mocap_data[prediction_range][:, 0, :15], next_frame_vae_output),
                                                      dim=-1)
                    test_history.copy_(next_frame_vae_output.detach())
                    history_input_guide.copy_(next_input_guide.detach())
                    prediction[step] = next_frame_vae_output.detach().cpu().numpy()
                    ep_total_loss_2 += float(total_loss)
                    ep_recon_loss_2 += float(recon_loss)
                    ep_z_mu_loss_2 += float(z_mu_loss)
                    ep_z_var_loss_2 += float(z_var_loss)

                prediction_dn = pose_vae.denormalize(prediction, if_np=1)
                prediction_dn_list.append(prediction_dn)
            prediction_dn_all = np.concatenate(prediction_dn_list, axis=0)
            prediction_dn_all_pos = prediction_dn_all[:, 3:30].reshape((-1, 9, 3))
            prediction_dn_all_rot = prediction_dn_all[:, 57:].reshape((-1, 9, 6))
            prediction_dn_all_rm = from_6D_to_rm(torch.from_numpy(prediction_dn_all_rot)).detach().cpu().numpy()
            prediction_dn_all_euler = np.zeros((len(prediction_dn_all), 9, 3))
            for joint in range(9):
                r = Rotation.from_matrix(prediction_dn_all_rm[:, joint])
                prediction_dn_all_euler[:, joint] = r.as_euler(bvh_order, degrees=True)
            avg_ep_recon_loss_phy_pos_2 = np.linalg.norm(mocap_data_ori_pos.detach().cpu().numpy() - prediction_dn_all_pos, axis=-1).mean()
            avg_ep_recon_loss_phy_rot_2 = np.abs(euler.detach().cpu().numpy() - prediction_dn_all_euler).mean()

            avg_ep_total_loss_2 = ep_total_loss_2 / count_steps
            avg_ep_recon_loss_2 = ep_recon_loss_2 / count_steps
            avg_ep_z_mu_loss_2 = ep_z_mu_loss_2 / count_steps
            avg_ep_z_var_loss_2 = ep_z_var_loss_2 / count_steps
            logger_curve.add_scalar('train/total_loss_2', avg_ep_total_loss_2, ep)
            logger_curve.add_scalar('train/recon_loss_2', avg_ep_recon_loss_2, ep)
            logger_curve.add_scalar('train/z_mu_loss_2', avg_ep_z_mu_loss_2, ep)
            logger_curve.add_scalar('train/z_var_loss_2', avg_ep_z_var_loss_2, ep)
            logger_curve.add_scalar('train/recon_loss_2_pos', avg_ep_recon_loss_phy_pos_2 / 1000, ep)
            logger_curve.add_scalar('train/recon_loss_2_rot', avg_ep_recon_loss_phy_rot_2, ep)

            #torch.save(copy.deepcopy(guide_network).cpu(), 'saved_models_v1/train_guide_low_1.pt')

            #test phase
            #pose_vae.eval()
            test_ep_total_loss_2 = 0
            test_ep_recon_loss_2 = 0
            test_ep_z_mu_loss_2 = 0
            test_ep_z_var_loss_2 = 0
            test_count_steps = 0
            test_prediction_list = []
            test_prediction_dn_list = []
            for i in range(test_num_peds):
                if i == 0:
                    test_num_steps = test_end_indices[0] + 1
                    test_condition_range = torch.tensor([[0]])
                else:
                    test_num_steps = test_end_indices[i] - test_end_indices[i - 1]
                    test_condition_range = torch.tensor(test_end_indices[i - 1]).reshape(1,1).long() + 1
                test_count_steps += test_num_steps - 1
                test_history.copy_(test_mocap_data[test_condition_range][:,0,15:])
                test_history_input_guide = test_mocap_data[test_condition_range].reshape((1, -1))
                test_prediction = torch.zeros((test_num_steps, 111)).to(args.device)
                test_prediction[0] = copy.deepcopy(test_history.squeeze())
                for step in range(1, test_num_steps):
                    test_prediction_range = (test_condition_range
                                        + torch.arange(step, step + 1).long()
                                        )
                    test_ground_truth = test_mocap_data[test_prediction_range][:,0,15:]
                    test_gt_z, test_gt_z_mu, test_gt_z_var = pose_vae.encode(test_ground_truth, test_history)
                    test_latent_z, test_latent_z_mu, test_latent_z_var = guide_network(test_history_input_guide[:, :15], test_history_input_guide[:, 15:])

                    test_next_frame_vae_output, test_recon_loss = \
                        feed_vae_eval(pose_vae, test_ground_truth, test_history, 1, test_latent_z)

                    test_z_mu_loss = (test_latent_z_mu - test_gt_z_mu).pow(2).mean()
                    test_z_var_loss = (test_latent_z_var - test_gt_z_var).pow(2).mean()
                    test_total_loss = test_z_mu_loss + test_z_var_loss

                    test_next_input_guide = torch.cat((test_mocap_data[test_prediction_range][:,0,:15], test_next_frame_vae_output), dim=-1)

                    test_history.copy_(test_next_frame_vae_output.detach())
                    test_history_input_guide.copy_(test_next_input_guide.detach())
                    test_prediction[step] = test_next_frame_vae_output
                    test_ep_total_loss_2 += float(test_total_loss)
                    test_ep_recon_loss_2 += float(test_recon_loss)
                    test_ep_z_mu_loss_2 += float(test_z_mu_loss)
                    test_ep_z_var_loss_2 += float(test_z_var_loss)

                test_prediction_dn = pose_vae.denormalize(test_prediction)
                test_prediction_list.append(test_prediction)
                test_prediction_dn_list.append(test_prediction_dn)
            test_prediction_dn_all = torch.cat(test_prediction_dn_list, dim=0)
            test_prediction_dn_all_pos = test_prediction_dn_all[:, 3:30].reshape((-1, 9, 3))
            test_prediction_dn_all_rot = test_prediction_dn_all[:, 57:].reshape((-1, 9, 6))
            test_prediction_dn_all_rm = from_6D_to_rm(test_prediction_dn_all_rot).cpu().detach().numpy()
            test_prediction_dn_all_euler = np.zeros((len(test_prediction_dn_all), 9, 3))
            for joint in range(9):
                r = Rotation.from_matrix(test_prediction_dn_all_rm[:, joint])
                test_prediction_dn_all_euler[:, joint] = r.as_euler(bvh_order, degrees=True)
            test_prediction_dn_all_euler = torch.from_numpy(test_prediction_dn_all_euler)
            test_avg_ep_recon_loss_phy_pos_2 = torch.norm(test_mocap_data_ori_pos - test_prediction_dn_all_pos,
                                                          dim=-1).mean()
            test_avg_ep_recon_loss_phy_rot_2 = torch.abs(test_euler - test_prediction_dn_all_euler).mean()

            test_avg_ep_total_loss_2 = test_ep_total_loss_2 / test_count_steps
            test_avg_ep_recon_loss_2 = test_ep_recon_loss_2 / test_count_steps
            test_avg_ep_z_mu_loss_2 = test_ep_z_mu_loss_2 / test_count_steps
            test_avg_ep_z_var_loss_2 = test_ep_z_var_loss_2 / test_count_steps
            logger_curve.add_scalar('test/total_loss_2', test_avg_ep_total_loss_2, ep)
            logger_curve.add_scalar('test/recon_loss_2', test_avg_ep_recon_loss_2, ep)
            logger_curve.add_scalar('test/z_mu_loss_2', test_avg_ep_z_mu_loss_2, ep)
            logger_curve.add_scalar('test/z_var_loss_2', test_avg_ep_z_var_loss_2, ep)
            logger_curve.add_scalar('test/recon_loss_2_pos', test_avg_ep_recon_loss_phy_pos_2 / 1000, ep)
            logger_curve.add_scalar('test/recon_loss_2_rot', test_avg_ep_recon_loss_phy_rot_2, ep)
            if test_avg_ep_recon_loss_phy_pos_2 < best_test_loss:
                print(f'epoch:{ep} per epoch average testing loss improves from {best_test_loss} to {test_avg_ep_recon_loss_phy_pos_2}')
                print('current_rec_loss:', test_avg_ep_recon_loss_2)
                best_test_loss = test_avg_ep_recon_loss_phy_pos_2
                torch.save(copy.deepcopy(guide_network).cpu(), 'saved_models/guideLow.pt')
                #torch.save(copy.deepcopy(pose_vae).cpu(), 'saved_models/mvae_guide_low_2.pt')

            # testp_ep_total_loss_2 = 0
            # testp_ep_recon_loss_2 = 0
            # testp_ep_z_mu_loss_2 = 0
            # testp_ep_z_var_loss_2 = 0
            # testp_count_steps = 0
            # testp_prediction_list = []
            # testp_prediction_dn_list = []
            # for i in range(testp_num_peds):
            #     if i == 0:
            #         testp_num_steps = testp_end_indices[0] + 1
            #         testp_condition_range = torch.tensor([[0]])
            #     else:
            #         testp_num_steps = testp_end_indices[i] - testp_end_indices[i - 1]
            #         testp_condition_range = torch.tensor(testp_end_indices[i - 1]).reshape(1, 1).long() + 1
            #     testp_count_steps += testp_num_steps - 1
            #     testp_history.copy_(testp_mocap_data[testp_condition_range][:, 0, 15:])
            #     testp_history_input_guide = testp_mocap_data[testp_condition_range].reshape((1, -1))
            #     testp_prediction = torch.zeros((testp_num_steps, 111)).to(args.device)
            #     testp_prediction[0] = copy.deepcopy(testp_history.squeeze())
            #     for step in range(1, testp_num_steps):
            #         testp_prediction_range = (testp_condition_range
            #                                  + torch.arange(step, step + 1).long()
            #                                  )
            #         testp_ground_truth = testp_mocap_data[testp_prediction_range][:, 0, 15:]
            #         testp_gt_z, testp_gt_z_mu, testp_gt_z_var = pose_vae.encode(testp_ground_truth, testp_history)
            #         testp_latent_z, testp_latent_z_mu, testp_latent_z_var = guide_network(testp_history_input_guide[:, :15],
            #                                                                            testp_history_input_guide[:, 15:])
            #
            #         testp_next_frame_vae_output, testp_recon_loss = \
            #             feed_vae_eval(pose_vae, testp_ground_truth, testp_history, 1, testp_latent_z)
            #
            #         testp_z_mu_loss = (testp_latent_z_mu - testp_gt_z_mu).pow(2).mean()
            #         testp_z_var_loss = (testp_latent_z_var - testp_gt_z_var).pow(2).mean()
            #
            #         testp_total_loss = testp_z_mu_loss + testp_z_var_loss
            #
            #         testp_next_input_guide = torch.cat(
            #             (testp_mocap_data[testp_prediction_range][:, 0, :15], testp_next_frame_vae_output), dim=-1)
            #
            #         testp_history.copy_(testp_next_frame_vae_output.detach())
            #         testp_history_input_guide.copy_(testp_next_input_guide.detach())
            #         testp_prediction[step] = testp_next_frame_vae_output
            #         testp_ep_total_loss_2 += float(testp_total_loss)
            #         testp_ep_recon_loss_2 += float(testp_recon_loss)
            #         testp_ep_z_mu_loss_2 += float(testp_z_mu_loss)
            #         testp_ep_z_var_loss_2 += float(testp_z_var_loss)
            #
            #     testp_prediction_dn = pose_vae.denormalize(testp_prediction)
            #     testp_prediction_list.append(testp_prediction)
            #     testp_prediction_dn_list.append(testp_prediction_dn)
            # testp_prediction_dn_all = torch.cat(testp_prediction_dn_list, dim=0)
            # testp_prediction_dn_all_pos = testp_prediction_dn_all[:, 3:30].reshape((-1, 9, 3))
            # testp_prediction_dn_all_rot = testp_prediction_dn_all[:, 57:].reshape((-1, 9, 6))
            # testp_prediction_dn_all_rm = from_6D_to_rm(testp_prediction_dn_all_rot).cpu().detach().numpy()
            # testp_prediction_dn_all_euler = np.zeros((len(testp_prediction_dn_all), 9, 3))
            # for joint in range(9):
            #     r = Rotation.from_matrix(testp_prediction_dn_all_rm[:, joint])
            #     testp_prediction_dn_all_euler[:, joint] = r.as_euler(bvh_order, degrees=True)
            # testp_prediction_dn_all_euler = torch.from_numpy(testp_prediction_dn_all_euler)
            # testp_avg_ep_recon_loss_phy_pos_2 = torch.norm(testp_mocap_data_ori_pos - testp_prediction_dn_all_pos,
            #                                               dim=-1).mean()
            # testp_avg_ep_recon_loss_phy_rot_2 = torch.abs(testp_euler - testp_prediction_dn_all_euler).mean()
            #
            # testp_avg_ep_total_loss_2 = testp_ep_total_loss_2 / testp_count_steps
            # testp_avg_ep_recon_loss_2 = testp_ep_recon_loss_2 / testp_count_steps
            # testp_avg_ep_z_mu_loss_2 = testp_ep_z_mu_loss_2 / testp_count_steps
            # testp_avg_ep_z_var_loss_2 = testp_ep_z_var_loss_2 / testp_count_steps
            # logger_curve.add_scalar('testp/total_loss_2', testp_avg_ep_total_loss_2, ep)
            # logger_curve.add_scalar('testp/recon_loss_2', testp_avg_ep_recon_loss_2, ep)
            # logger_curve.add_scalar('testp/z_mu_loss_2', testp_avg_ep_z_mu_loss_2, ep)
            # logger_curve.add_scalar('testp/z_var_loss_2', testp_avg_ep_z_var_loss_2, ep)
            # logger_curve.add_scalar('testp/recon_loss_2_pos', testp_avg_ep_recon_loss_phy_pos_2 / 1000, ep)
            # logger_curve.add_scalar('testp/recon_loss_2_rot', testp_avg_ep_recon_loss_phy_rot_2, ep)
            # if testp_avg_ep_recon_loss_phy_pos_2 < best_testp_loss:
            #     print(
            #         f'epoch:{ep} per epoch average testing loss prediction_version improves from {best_testp_loss} to {testp_avg_ep_recon_loss_phy_pos_2}')
            #     print('current_rec_loss:', testp_avg_ep_recon_loss_2)
            #     best_testp_loss = testp_avg_ep_recon_loss_phy_pos_2
            #     torch.save(copy.deepcopy(guide_network).cpu(), 'saved_models/testp_guide_low_2.pt')
            #     torch.save(copy.deepcopy(pose_vae).cpu(), 'saved_models/mvaep_guide_low_2.pt')
        guide_network.train()

if __name__ == "__main__":
    main()