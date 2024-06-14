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

from common.logging_utils import CSVLogger
from common.misc_utils import update_linear_schedule
from models import (
    PoseMixtureVAE_up
)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def from_6D_to_rm(orientation_6D):
    a1, a2 = orientation_6D[..., [0, 2, 4]], orientation_6D[..., [1, 3, 5]]
    b1 = F.normalize(a1, dim=-1)
    # diff = a1 - b1
    # diff_norm = torch.abs(diff).sum()
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1,b2,b3), dim=3)
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
        ep_perplexity = data["ep_perplexity"]

        print(
            "{} | Recon: {:.3e} | KL: {:.3e} | PP: {:.3e}".format(
                self.time_since(ep), ep_recon_loss, ep_kl_loss, ep_perplexity
            ),
            flush=True,
        )
def feed_vae_eval(pose_vae, ground_truth, condition, future_weights, device):
    condition = condition.flatten(start_dim=1, end_dim=2)

    output_shape = (-1, pose_vae.num_future_predictions, pose_vae.frame_size_rec)

    z = torch.randn(1, 64).to(device)
    #z = torch.zeros((1,32))
    vae_output = pose_vae.sample(z, condition)
    vae_output = vae_output.view(output_shape)

    recon_loss = (vae_output - ground_truth).pow(2).mean(dim=(0, -1))
    recon_loss = recon_loss.mul(future_weights).sum()

    return vae_output, recon_loss

def feed_vae(pose_vae, ground_truth, condition, future_weights):
    condition = condition.flatten(start_dim=1, end_dim=2)
    flattened_truth = ground_truth.flatten(start_dim=1, end_dim=2)

    output_shape = (-1, pose_vae.num_future_predictions, pose_vae.frame_size_rec)

    # PoseVAE and PoseMixtureVAE
    vae_output, mu, logvar = pose_vae(flattened_truth, condition)
    vae_output = vae_output.view(output_shape)

    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum().clamp(max=0)
    kl_loss /= logvar.numel()

    recon_loss = (vae_output - ground_truth).pow(2).mean(dim=(0, -1))
    recon_loss = recon_loss.mul(future_weights).sum()

    return (vae_output, mu, logvar), (recon_loss, kl_loss)


def main():
    print('cuda:2, 0.001')
    logger_curve = SummaryWriter('runs/3')
    env_path = os.path.join(parent_dir, "environments")

    # setup parameters
    args = SimpleNamespace(
        device="cuda:2" if torch.cuda.is_available() else "cpu",
        mocap_file=os.path.join(env_path, "mocap.npz"),
        norm_mode="zscore",
        latent_size=64,
        num_embeddings=12,
        num_experts=4,
        num_condition_frames=1,
        num_future_predictions=1,
        num_steps_per_rollout=32,
        kl_beta=0.001,
        load_saved_model=False,
    )

    # learning parameters
    teacher_epochs = 20
    ramping_epochs = 20
    student_epochs = 100
    args.num_epochs = teacher_epochs + ramping_epochs + student_epochs
    #args.num_epochs = 1000
    args.mini_batch_size = 64
    args.initial_lr = 1e-4
    args.final_lr = 1e-7

    with open('processed_data/train_data_up_dyn.pkl', 'rb') as f:
        raw_data = pickle.load(f)
    mocap_data_ori = torch.from_numpy(raw_data["data"]).float().to(args.device)
    end_indices = raw_data["end_indices"]
    bvh_order = 'zyx'

    mocap_data_ori_facing = mocap_data_ori[:, 2] * 180 / torch.pi
    mocap_data_ori_pos = mocap_data_ori[:, 3:45].reshape((-1, 14, 3))
    mocap_data_ori_vel = mocap_data_ori[:, 45:87].reshape((-1, 14, 3))
    mocap_data_ori_rot = mocap_data_ori[:, 87:171].reshape((-1, 14, 6))
    mocap_data_ori_rm = from_6D_to_rm(mocap_data_ori_rot).cpu().detach().numpy()
    euler = np.zeros((len(mocap_data_ori), 14, 3))
    for joint in range(14):
        r = Rotation.from_matrix(mocap_data_ori_rm[:, joint])
        euler[:, joint] = r.as_euler(bvh_order, degrees=True)
    euler = torch.from_numpy(euler)

    max = mocap_data_ori.max(dim=0)[0]
    min = mocap_data_ori.min(dim=0)[0]
    avg = mocap_data_ori.mean(dim=0)
    std = mocap_data_ori.std(dim=0)

    # Make sure we don't divide by 0
    std[std == 0] = 1.0

    normalization = {
        "mode": args.norm_mode,
        "max": max,
        "min": min,
        "avg": avg,
        "std": std,
    }


    if args.norm_mode == "zscore":
        mocap_data = (mocap_data_ori - avg) / std
    mocap_data_rec = mocap_data[:,:171]

    batch_size = mocap_data.size()[0]
    frame_size_con = mocap_data.size()[1]
    frame_size_rec = mocap_data_rec.size()[1]

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

    pose_vae = PoseMixtureVAE_up(
        frame_size_con,
        frame_size_rec,
        args.latent_size,
        args.num_condition_frames,
        args.num_future_predictions,
        normalization,
        args.num_experts,
    ).to(args.device)

    pose_vae.train()

    vae_optimizer = optim.Adam(pose_vae.parameters(), lr=args.initial_lr)

    sample_schedule = torch.cat(
        (
            # First part is pure teacher forcing
            torch.zeros(teacher_epochs),
            # Second part with schedule sampling
            torch.linspace(0.0, 1.0, ramping_epochs),
            # last part is pure student
            torch.ones(student_epochs),
        )
    )
    #sample_schedule = torch.ones(1000)
    # future_weights = torch.softmax(
    #     torch.linspace(1, 0, args.num_future_predictions), dim=0
    # ).to(args.device)

    future_weights = (
        torch.ones(args.num_future_predictions)
        .to(args.device)
        .div_(args.num_future_predictions)
    )
    num_peds = len(end_indices)

    # buffer for later
    shape = (args.mini_batch_size, args.num_condition_frames, frame_size_con)
    history = torch.empty(shape).to(args.device)

    log_path = os.path.join(current_dir, "log_posevae_progress")
    logger = StatsLogger(args, csv_path=log_path)

    with open('processed_data/test_data_up_dyn.pkl', 'rb') as f:
        test_raw_data = pickle.load(f)
    test_mocap_data_ori = torch.from_numpy(test_raw_data["data"]).float().to(args.device)
    test_end_indices = test_raw_data["end_indices"]
    # with open('processed_data/pred_lower.pkl', 'rb') as f:
    #     test_raw_lower_data_list = pickle.load(f)
    # feature_len = test_raw_lower_data_list[0].size()[1]
    # padding_vec = torch.zeros((1, feature_len))
    # for j_pred in range(len(test_raw_lower_data_list)):
    #     test_raw_lower_data_list[j_pred] = torch.cat((test_raw_lower_data_list[j_pred], padding_vec), dim=0)[1:]
    # test_pred_lower_data = torch.cat(test_raw_lower_data_list, dim=0)
    # test_mocap_data[:, 171:] = test_pred_lower_data[:, 3:30]


    if args.norm_mode == "zscore":
        test_mocap_data = (test_mocap_data_ori - avg) / std

    test_mocap_data_ori_facing = test_mocap_data_ori[:, 2] * 180 / torch.pi
    test_mocap_data_ori_pos = test_mocap_data_ori[:, 3:45].reshape((-1, 14, 3))
    test_mocap_data_ori_vel = test_mocap_data_ori[:, 45:87].reshape((-1, 14, 3))
    test_mocap_data_ori_rot = test_mocap_data_ori[:, 87:171].reshape((-1, 14, 6))
    test_mocap_data_ori_rm = from_6D_to_rm(test_mocap_data_ori_rot).cpu().detach().numpy()
    test_euler = np.zeros((len(test_mocap_data_ori), 14, 3))
    for joint in range(14):
        test_r = Rotation.from_matrix(test_mocap_data_ori_rm[:, joint])
        test_euler[:, joint] = test_r.as_euler(bvh_order, degrees=True)
    test_euler = torch.from_numpy(test_euler)

    test_mocap_data_rec = test_mocap_data[:,:171]
    test_num_peds = len(test_end_indices)
    test_shape = (1, args.num_condition_frames, frame_size_con)
    test_history = torch.empty(test_shape).to(args.device)
    best_test_loss = np.inf
    for ep in range(1, args.num_epochs + 1):
        sampler = BatchSampler(
            SubsetRandomSampler(selectable_indices),
            args.mini_batch_size,
            drop_last=True,
        )
        ep_recon_loss = 0
        ep_recon_loss_phy_facing = 0
        ep_recon_loss_phy_pos = 0
        ep_recon_loss_phy_vel = 0
        ep_recon_loss_phy_rot = 0
        ep_kl_loss = 0
        ep_perplexity = 0

        update_linear_schedule(
            vae_optimizer, ep - 1, args.num_epochs, args.initial_lr, args.final_lr
        )

        for num_mini_batch, indices in enumerate(sampler):
            t_indices = torch.LongTensor(indices)

            # condition is from newest...oldest, i.e. (t-1, t-2, ... t-n)
            condition_range = (
                t_indices.repeat((args.num_condition_frames, 1)).t()
                + torch.arange(args.num_condition_frames - 1, -1, -1).long()
            )

            t_indices += args.num_condition_frames
            history[:, : args.num_condition_frames].copy_(mocap_data[condition_range])

            for offset in range(args.num_steps_per_rollout):
                # dims: (num_parallel, num_window, feature_size)
                use_student = torch.rand(1) < sample_schedule[ep - 1]

                prediction_range = (
                    t_indices.repeat((args.num_future_predictions, 1)).t()
                    + torch.arange(offset, offset + args.num_future_predictions).long()
                )
                ground_truth = mocap_data_rec[prediction_range]
                ground_truth_con = mocap_data[prediction_range]
                condition = history[:, : args.num_condition_frames]

                gt_facing = mocap_data_ori_facing[prediction_range].squeeze()
                gt_pos = mocap_data_ori_pos[prediction_range].squeeze()
                gt_vel = mocap_data_ori_vel[prediction_range].squeeze()
                gt_euler = euler[prediction_range].squeeze()


                # PoseVAE, PoseMixtureVAE, PoseMixtureSpecialistVAE
                (vae_output, _, _), (recon_loss, kl_loss) = feed_vae(
                    pose_vae, ground_truth, condition, future_weights
                )

                history = history.roll(1, dims=1)
                if use_student:
                    next_frame_up = vae_output[:, 0]
                    next_frame = torch.cat((next_frame_up, ground_truth_con.squeeze()[:, 171:]), dim=-1)
                else:
                    next_frame = ground_truth_con[:, 0]

                history[:, 0].copy_(next_frame.detach())

                vae_optimizer.zero_grad()
                (recon_loss + args.kl_beta * kl_loss).backward()
                vae_optimizer.step()

                ep_recon_loss += float(recon_loss) / args.num_steps_per_rollout
                ep_kl_loss += float(kl_loss) / args.num_steps_per_rollout

                # vae_output_dn = pose_vae.denormalize(vae_output.detach()[:, 0], if_rec=True)
                # pred_facing = vae_output_dn[:, 2]
                # pred_pos = vae_output_dn[:, 3:45].reshape((-1, 14, 3))
                # pred_vel = vae_output_dn[:, 45:87].reshape((-1, 14, 3))
                # pred_rot = vae_output_dn[:, 87:171].reshape((-1, 14, 6))
                # pred_rm = from_6D_to_rm(pred_rot).cpu().detach().numpy()
                # pred_euler = np.zeros((len(pred_rm), 14, 3))
                # for joint in range(14):
                #     r = Rotation.from_matrix(pred_rm[:, joint])
                #     pred_euler[:, joint] = r.as_euler(bvh_order, degrees=True)
                # pred_euler = torch.from_numpy(pred_euler)

                # ep_recon_loss_phy_facing += torch.abs(gt_facing - pred_facing).mean() / args.num_steps_per_rollout
                # ep_recon_loss_phy_pos += torch.norm(gt_pos - pred_pos, dim=-1).mean() / args.num_steps_per_rollout
                # ep_recon_loss_phy_vel += torch.norm(gt_vel - pred_vel, dim=-1).mean() / args.num_steps_per_rollout
                # ep_recon_loss_phy_rot += torch.abs(gt_euler - pred_euler).mean() / args.num_steps_per_rollout


        avg_ep_recon_loss = ep_recon_loss / num_mini_batch
        avg_ep_kl_loss = ep_kl_loss / num_mini_batch
        # avg_ep_recon_loss_phy_facing = ep_recon_loss_phy_facing / num_mini_batch
        # avg_ep_recon_loss_phy_pos = ep_recon_loss_phy_pos / num_mini_batch
        # avg_ep_recon_loss_phy_vel = ep_recon_loss_phy_vel / num_mini_batch
        # avg_ep_recon_loss_phy_rot = ep_recon_loss_phy_rot / num_mini_batch
        # logger_curve.add_scalar('train/recon_loss_1', avg_ep_recon_loss, ep)
        # logger_curve.add_scalar('train/kl_loss', avg_ep_kl_loss, ep)
        # logger_curve.add_scalar('train/recon_loss_1_facing', avg_ep_recon_loss_phy_facing, ep)
        # logger_curve.add_scalar('train/recon_loss_1_pos', avg_ep_recon_loss_phy_pos / 1000, ep)
        # logger_curve.add_scalar('train/recon_loss_1_vel', avg_ep_recon_loss_phy_vel / 1000, ep)
        # logger_curve.add_scalar('train/recon_loss_1_rot', avg_ep_recon_loss_phy_rot, ep)


        logger.log_stats(
            {
                "epoch": ep,
                "ep_recon_loss": avg_ep_recon_loss,
                "ep_kl_loss": avg_ep_kl_loss,
                "ep_perplexity": 0,
            }
        )

        #torch.save(copy.deepcopy(pose_vae).cpu(), pose_vae_path)

        #test phase
        # pose_vae.eval()
        # ep_recon_loss_2 = 0
        # count_steps = 0
        # prediction_dn_list = []
        # for i in range(num_peds):
        #     if i == 0:
        #         num_steps = end_indices[0] + 1
        #         condition_range = torch.tensor([[0]])
        #     else:
        #         num_steps = end_indices[i] - end_indices[i - 1]
        #         condition_range = torch.tensor(end_indices[i - 1]).reshape(1,1).long() + 1
        #     count_steps += num_steps - 1
        #     test_history[:, : args.num_condition_frames].copy_(mocap_data[condition_range])
        #     prediction = torch.zeros((num_steps, 171)).to(args.device)
        #     prediction[0] = copy.deepcopy(test_history.squeeze()[:171])
        #     for step in range(1, num_steps):
        #         prediction_range = (condition_range
        #                             + torch.arange(step, step + 1).long()
        #                             )
        #         ground_truth = mocap_data_rec[prediction_range]
        #         ground_truth_con = mocap_data[prediction_range]
        #         condition = test_history[:, : args.num_condition_frames]
        #         (vae_output, _, _), (recon_loss, kl_loss) = feed_vae(
        #             pose_vae, ground_truth, condition, future_weights
        #         )
        #
        #         test_history = test_history.roll(1, dims=1)
        #         next_frame_up = vae_output[:, 0]
        #         next_frame = torch.cat((next_frame_up, ground_truth_con[:, 0, 171:],), dim=-1)
        #         test_history[:, 0].copy_(next_frame.detach())
        #
        #         ep_recon_loss_2 += float(recon_loss)
        #         prediction[step] = vae_output.detach()[:, 0]
        #     prediction_dn = pose_vae.denormalize(prediction, if_rec=True)
        #     prediction_dn_list.append(prediction_dn)
        # prediction_dn_all = torch.cat(prediction_dn_list, dim=0)
        # prediction_dn_all_facing = prediction_dn_all[:, 2] * 180 / torch.pi
        # prediction_dn_all_pos = prediction_dn_all[:, 3:45].reshape((-1, 14, 3))
        # prediction_dn_all_vel = prediction_dn_all[:, 45:87].reshape((-1, 14, 3))
        # prediction_dn_all_rot = prediction_dn_all[:, 87:].reshape((-1, 14, 6))
        # prediction_dn_all_rm = from_6D_to_rm(prediction_dn_all_rot).cpu().detach().numpy()
        # prediction_dn_all_euler = np.zeros((len(prediction_dn_all), 14, 3))
        # for joint in range(14):
        #     r = Rotation.from_matrix(prediction_dn_all_rm[:, joint])
        #     prediction_dn_all_euler[:, joint] = r.as_euler(bvh_order, degrees=True)
        # prediction_dn_all_euler = torch.from_numpy(prediction_dn_all_euler)
        # avg_ep_recon_loss_phy_facing_2 = torch.abs(mocap_data_ori_facing - prediction_dn_all_facing).mean()
        # avg_ep_recon_loss_phy_pos_2 = torch.norm(mocap_data_ori_pos - prediction_dn_all_pos, dim=-1).mean()
        # avg_ep_recon_loss_phy_vel_2 = torch.norm(mocap_data_ori_vel - prediction_dn_all_vel, dim=-1).mean()
        # avg_ep_recon_loss_phy_rot_2 = torch.abs(euler - prediction_dn_all_euler).mean()

        # avg_ep_recon_loss_2 = ep_recon_loss_2 / count_steps
        # logger_curve.add_scalar('train/recon_loss_2', avg_ep_recon_loss_2, ep)
        # logger_curve.add_scalar('train/recon_loss_2_facing', avg_ep_recon_loss_phy_facing_2, ep)
        # logger_curve.add_scalar('train/recon_loss_2_pos', avg_ep_recon_loss_phy_pos_2 / 1000, ep)
        # logger_curve.add_scalar('train/recon_loss_2_vel', avg_ep_recon_loss_phy_vel_2 / 1000, ep)
        # logger_curve.add_scalar('train/recon_loss_2_rot', avg_ep_recon_loss_phy_rot_2, ep)


        test_ep_recon_loss_2 = 0
        test_count_steps = 0
        test_prediction_dn_list = []
        for i in range(test_num_peds):
            if i == 0:
                test_num_steps = test_end_indices[0] + 1
                test_condition_range = torch.tensor([[0]])
            else:
                test_num_steps = test_end_indices[i] - test_end_indices[i - 1]
                test_condition_range = torch.tensor(test_end_indices[i - 1]).reshape(1,1).long() + 1
            test_count_steps += test_num_steps - 1
            test_history[:, : args.num_condition_frames].copy_(test_mocap_data[test_condition_range])
            test_prediction = torch.zeros((test_num_steps, 171)).to(args.device)
            test_prediction[0] = copy.deepcopy(test_history.squeeze()[:171])
            for step in range(1, test_num_steps):
                test_prediction_range = (test_condition_range
                                    + torch.arange(step, step + 1).long()
                                    )
                test_ground_truth = test_mocap_data_rec[test_prediction_range]
                test_ground_truth_con = test_mocap_data[test_prediction_range]
                test_condition = test_history[:, : args.num_condition_frames]
                (test_vae_output, _, _), (test_recon_loss, test_kl_loss) = feed_vae(
                    pose_vae, test_ground_truth, test_condition, future_weights
                )

                test_history = test_history.roll(1, dims=1)
                test_next_frame_up = test_vae_output[:, 0]
                test_next_frame = torch.cat((test_next_frame_up, test_ground_truth_con[:, 0, 171:],), dim=-1)
                test_history[:, 0].copy_(test_next_frame.detach())

                test_ep_recon_loss_2 += float(test_recon_loss)
                test_prediction[step] = test_vae_output.detach()[:, 0]
            test_prediction_dn = pose_vae.denormalize(test_prediction, if_rec=True)
            test_prediction_dn_list.append(test_prediction_dn)
        test_prediction_dn_all = torch.cat(test_prediction_dn_list, dim=0)
        # test_prediction_dn_all_facing = test_prediction_dn_all[:, 2] * 180 / torch.pi
        test_prediction_dn_all_pos = test_prediction_dn_all[:, 3:45].reshape((-1, 14, 3))
        # test_prediction_dn_all_vel = test_prediction_dn_all[:, 45:87].reshape((-1, 14, 3))
        # test_prediction_dn_all_rot = test_prediction_dn_all[:, 87:].reshape((-1, 14, 6))
        # test_prediction_dn_all_rm = from_6D_to_rm(test_prediction_dn_all_rot).cpu().detach().numpy()
        # test_prediction_dn_all_euler = np.zeros((len(test_prediction_dn_all), 14, 3))
        # for joint in range(14):
        #     r = Rotation.from_matrix(test_prediction_dn_all_rm[:, joint])
        #     test_prediction_dn_all_euler[:, joint] = r.as_euler(bvh_order, degrees=True)
        # test_prediction_dn_all_euler = torch.from_numpy(test_prediction_dn_all_euler)
        # test_avg_ep_recon_loss_phy_facing_2 = torch.abs(test_mocap_data_ori_facing - test_prediction_dn_all_facing).mean()
        test_avg_ep_recon_loss_phy_pos_2 = torch.norm(test_mocap_data_ori_pos - test_prediction_dn_all_pos, dim=-1).mean()
        # test_avg_ep_recon_loss_phy_vel_2 = torch.norm(test_mocap_data_ori_vel - test_prediction_dn_all_vel, dim=-1).mean()
        # test_avg_ep_recon_loss_phy_rot_2 = torch.abs(test_euler - test_prediction_dn_all_euler).mean()

        test_avg_ep_recon_loss_2 = test_ep_recon_loss_2 / test_count_steps
        logger_curve.add_scalar('test/recon_loss_2', test_avg_ep_recon_loss_2, ep)
        # logger_curve.add_scalar('test/recon_loss_2_facing', test_avg_ep_recon_loss_phy_facing_2, ep)
        # logger_curve.add_scalar('test/recon_loss_2_pos', test_avg_ep_recon_loss_phy_pos_2 / 1000, ep)
        # logger_curve.add_scalar('test/recon_loss_2_vel', test_avg_ep_recon_loss_phy_vel_2 / 1000, ep)
        # logger_curve.add_scalar('test/recon_loss_2_rot', test_avg_ep_recon_loss_phy_rot_2, ep)

        if test_avg_ep_recon_loss_phy_pos_2 < best_test_loss:
            print(f'epoch:{ep} per epoch average testing loss improves from {best_test_loss} to {test_avg_ep_recon_loss_phy_pos_2}')
            best_test_loss = test_avg_ep_recon_loss_phy_pos_2
            torch.save(copy.deepcopy(pose_vae).cpu(), 'saved_models/cvaeUp.pt')
        pose_vae.train()

if __name__ == "__main__":
    main()