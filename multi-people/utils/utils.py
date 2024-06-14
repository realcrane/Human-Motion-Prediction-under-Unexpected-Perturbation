import torch
import numpy as np
import torch.nn.functional as F
import time
import pdb
def calculate_mats(M, m, l, theta, phi, theta_dot, phi_dot, g, device='cpu'):

    #claculate mat_M
    mat_M = torch.zeros((4,4)).to(device)
    mat_M[0,0] = M + m
    mat_M[0,2] = m * l * torch.cos(theta)
    mat_M[1,1] = M + m
    mat_M[1,2] = m * l * torch.sin(theta) * torch.sin(phi)
    mat_M[1,3] = -m * l * torch.cos(theta) * torch.cos(phi)
    mat_M[2,2] = m * torch.pow(l,2)
    mat_M[3,3] = m * torch.pow(l,2) * torch.pow(torch.cos(theta),2)
    mat_M[2,0], mat_M[2,1], mat_M[3,1] = mat_M[0,2], mat_M[1,2], mat_M[1,3]


    #calculate mat_C
    mat_C = torch.zeros((4,1)).to(device)
    mat_C[0,0] = - m * l * torch.sin(theta) * torch.pow(theta_dot,2)
    mat_C[1,0] = 2 * m * l * torch.sin(theta) * torch.cos(phi) * theta_dot * phi_dot +  m * l * torch.cos(theta) * torch.sin(phi)*(torch.pow(theta_dot,2) + torch.pow(phi_dot,2))
    mat_C[2,0] = m * torch.pow(l,2) * torch.sin(theta) * torch.cos(theta) *  torch.pow(phi_dot,2)
    mat_C[3,0] = -2 * m * torch.pow(l,2) * torch.sin(theta) * torch.cos(theta) * theta_dot * phi_dot

    #claculate mat_G
    mat_G = torch.zeros((4, 1)).to(device)
    mat_G[2,0] = -m * g * l * torch.sin(theta) * torch.cos(phi)
    mat_G[3,0] = -m * g * l * torch.cos(theta) * torch.sin(phi)


    return mat_M.double(), mat_C.double(), mat_G.double()

# def PD_control(pd_params, cur_st, last_st, obj_st, device='cpu'):
#     cur_err = obj_st - cur_st
#     las_err = obj_st - last_st
#     keys = list(pd_params.keys())
#     out = torch.zeros(len(keys)).to(device)
#
#     for i in range(len(keys)):
#         out[i] = pd_params[keys[i]][0] * cur_err[i] + pd_params[keys[i]][1] * (cur_err[i] - las_err[ i])
#     return out

def generate_poses(ini_pos, positions):
    num_peds = ini_pos.shape[0]
    tran_vs = np.zeros((num_peds, 1, 3))
    tran_vs[:, 0, :2] = ini_pos
    ini_poses = np.repeat(positions, num_peds, axis=0) + tran_vs
    return ini_poses

def ipm_st2feature_low(ipm_motion, ini_poses, device = 'cpu'):
    if not torch.is_tensor(ini_poses):
        if type(ini_poses) is np.ndarray:
            ini_poses = torch.from_numpy(ini_poses).double().to(device)
        else:
            raise Exception('Unknown dtype')
    ini_poses_m = ini_poses / 1000
    num_peds = ipm_motion.shape[0]
    num_frames = ipm_motion.shape[1]
    tran_v = torch.zeros((num_peds, 1, 4)).double().to(device)
    tran_v[:, 0, :2] = ini_poses_m[:, 0, :2]
    ipm_motion_tran = ipm_motion - tran_v
    rod_len = cal_rod(ini_poses_m)
    ipm_motion_tran_dot = ipm_motion_tran[:, 1:] - ipm_motion_tran[:, :-1]

    cart_height = (ini_poses_m[:, 16, -1] + ini_poses_m[:, 20, -1]) / 2 # num_peds
    rod_vec = torch.ones((num_peds, num_frames)).double().to(device) * rod_len
    end_rod = torch.zeros((num_peds, num_frames, 3)).double().to(device)

    for i in range(num_peds):
        for j in range(num_frames):
            x, y, theta, phi = ipm_motion_tran[i, j, 0], ipm_motion_tran[i, j, 1], ipm_motion_tran[i, j, 2], \
                               ipm_motion_tran[i, j, 3]
            end_of_pendulum_local = rod_vec[i, j] * torch.tensor(
                [[torch.sin(theta)], [-torch.cos(theta) * torch.sin(phi)], [torch.cos(theta) * torch.cos(phi)]])
            end_of_pendulum = torch.tensor([[x], [y], [cart_height[i]]]) + end_of_pendulum_local
            end_rod[i, j, :] = end_of_pendulum.reshape(3)
    end_rod_v = end_rod[:, 1:] - end_rod[:, :-1]
    ipm_features = torch.cat((ipm_motion_tran[:, :-1], end_rod[:, :-1], rod_vec[:, :-1].unsqueeze(2),
                                                ipm_motion_tran_dot, end_rod_v), dim=-1)
    return ipm_features #num_peds * frames - 1 * 15
def ipm_st2feature_low_sgl(ipm_motion, ini_poses, device = 'cpu'):
    if not torch.is_tensor(ini_poses):
        if type(ini_poses) is np.ndarray:
            ini_poses = torch.from_numpy(ini_poses).double().to(device)
        else:
            raise Exception('Unknown dtype')
    ini_poses_m = ini_poses / 1000
    num_peds = ipm_motion.shape[0]
    num_frames = ipm_motion.shape[1]
    tran_v = torch.zeros((num_peds, 1, 4)).double().to(device)
    tran_v[:, 0, :2] = ini_poses_m[:, 0, :2]
    ipm_motion_tran = ipm_motion - tran_v
    rod_len = cal_rod(ini_poses_m)
    ipm_motion_tran_dot = ipm_motion_tran[:, 1:] - ipm_motion_tran[:, :-1]


    ipm_features = torch.cat((ipm_motion_tran[:, :-1, 2:], ipm_motion_tran_dot), dim=-1)
    return ipm_features #num_peds * frames - 1 * 6

def cal_rod(ini_poses_m):
    cart = (ini_poses_m[:, 16, :] + ini_poses_m[:, 20, :]) / 2
    hip = ini_poses_m[:, 0, :]
    rod_len = torch.norm((hip - cart), dim=-1).reshape(-1, 1) #num_peds * 1
    return rod_len

def ipm_st2feature_up(ipm_motion, ini_poses, device = 'cpu'):
    if not torch.is_tensor(ini_poses):
        if type(ini_poses) is np.ndarray:
            ini_poses = torch.from_numpy(ini_poses).double().to(device)
        else:
            raise Exception('Unknown dtype')
    ini_poses_m = ini_poses / 1000
    num_peds = ipm_motion.shape[0]
    tran_v = torch.zeros((num_peds, 1, 4)).double().to(device)
    tran_v[:, 0, :2] = ini_poses_m[:, 0, :2]
    ipm_motion_tran = ipm_motion - tran_v
    ipm_motion_tran_dot = ipm_motion_tran[:, 1:] - ipm_motion_tran[:, :-1]
    ipm_features = torch.cat((ipm_motion_tran[:, :-1, 2:], ipm_motion_tran_dot), dim=-1)

    return ipm_features #num_peds * frames - 1 * 6
def ipm_st2feature_up_sgl(ipm_motion, ini_poses, device = 'cpu'):
    if not torch.is_tensor(ini_poses):
        if type(ini_poses) is np.ndarray:
            ini_poses = torch.from_numpy(ini_poses).double().to(device)
        else:
            raise Exception('Unknown dtype')
    ini_poses_m = ini_poses / 1000
    num_peds = ipm_motion.shape[0]
    tran_v = torch.zeros((num_peds, 1, 4)).double().to(device)
    tran_v[:, 0, :2] = ini_poses_m[:, 0, :2]
    ipm_motion_tran = ipm_motion - tran_v
    ipm_motion_tran_dot = ipm_motion_tran[:, 1:] - ipm_motion_tran[:, :-1]
    ipm_features = torch.cat((ipm_motion_tran[:, :-1, 2:], ipm_motion_tran_dot), dim=-1)

    return ipm_features #num_peds * frames - 1 * 6

def ipm_st2fts(ipm_motions, ini_poses, stan_inputs_low, stan_inputs_up, gr, device):
    if gr:
        ipm_fts_ori_low = ipm_st2feature_low(ipm_motions, ini_poses, device)
        ipm_fts_ori_up = ipm_st2feature_up(ipm_motions, ini_poses, device)
        avg_low = stan_inputs_low['avg']
        std_low = stan_inputs_low['std']
        avg_up = stan_inputs_up['avg']
        std_up = stan_inputs_up['std']
    else:
        ipm_fts_ori_low = ipm_st2feature_low_sgl(ipm_motions, ini_poses, device)
        ipm_fts_ori_up = ipm_st2feature_up_sgl(ipm_motions, ini_poses, device)
        avg_low = stan_inputs_low['avg'][[2,3,8,9,10,11]]
        std_low = stan_inputs_low['std'][[2,3,8,9,10,11]]
        avg_up = stan_inputs_up['avg'][[2,3,8,9,10,11]]
        std_up = stan_inputs_up['std'][[2,3,8,9,10,11]]

    ipm_fts_dim_low = ipm_fts_ori_low.shape[-1]
    ipm_fts_dim_up = ipm_fts_ori_up.shape[-1]
    ipm_fts_low = (ipm_fts_ori_low - avg_low[:ipm_fts_dim_low].reshape(1, 1, ipm_fts_dim_low)) / std_low[:ipm_fts_dim_low].reshape(
        1, 1, ipm_fts_dim_low)
    ipm_fts_up = (ipm_fts_ori_up - avg_up[:ipm_fts_dim_up].reshape(1, 1, ipm_fts_dim_up)) / std_up[:ipm_fts_dim_up].reshape(
        1, 1, ipm_fts_dim_up)
    return ipm_fts_low, ipm_fts_up

def ipm_st2fts_low(ipm_motions, ini_poses, stan_inputs_low, device):
    ipm_fts_ori_low = ipm_st2feature_low(ipm_motions, ini_poses, device)
    ipm_fts_dim_low = ipm_fts_ori_low.shape[-1]

    avg_low = stan_inputs_low['avg']
    std_low = stan_inputs_low['std']
    ipm_fts_low = (ipm_fts_ori_low - avg_low[:ipm_fts_dim_low].reshape(1, 1, ipm_fts_dim_low)) / std_low[:ipm_fts_dim_low].reshape(
        1, 1, ipm_fts_dim_low)
    return ipm_fts_low

def tran_pose(ini_poses, num_peds, device):
    if not torch.is_tensor(ini_poses):
        if type(ini_poses) is np.ndarray:
            ini_poses = torch.from_numpy(ini_poses).double().to(device)
        else:
            raise Exception('Unknown dtype')
    tran_v_pos = torch.zeros((num_peds, 1, 3)).double().to(device)
    tran_v_pos[:, 0, :2] = ini_poses[:, 0, :2]
    ini_poses_tran = ini_poses - tran_v_pos
    return ini_poses_tran, tran_v_pos

def ft_cal_low(ini_poses_tran, ini_orientation, ipm_fts_dim, stan_inputs, num_peds, ft_len, num_joints, low_body, device='cpu'):
    ft = torch.zeros((num_peds, ft_len)).double().to(device)
    ft[:, 3: 3 + 3 * num_joints] = ini_poses_tran[:, low_body].reshape(num_peds, 3*num_joints)
    ft[:, 3 + 3 * num_joints * 2:] = ini_orientation[:, low_body, :, :].reshape(num_peds, num_joints, 6).reshape(num_peds, num_joints * 6)
    avg = stan_inputs['avg']
    std = stan_inputs['std']
    ft = (ft - avg[ipm_fts_dim:].reshape(1, -1)) / std[ipm_fts_dim:].reshape(1, -1).float()
    return ft

def ft_cal_up(ini_poses_tran, ini_orientation, ipm_fts_dim, stan_inputs, num_peds, ft_len_vae, num_joints,
              up_body,  device='cpu'):
    ft = torch.zeros((num_peds, ft_len_vae)).double().to(device)
    ft[:, 3: 3 + 3 * num_joints] = ini_poses_tran[:, up_body].reshape(num_peds, 3*num_joints)
    ft[:, 3 + 3 * num_joints * 2 : ] = ini_orientation[:, up_body, :, :].reshape(num_peds, num_joints, 6).reshape(num_peds, num_joints * 6)
    avg = stan_inputs['avg']
    std = stan_inputs['std']
    ft = (ft - avg[ipm_fts_dim:ipm_fts_dim+ft_len_vae].reshape(1, -1)) / std[ipm_fts_dim:ipm_fts_dim+ft_len_vae].reshape(1, -1).float()
    return ft


def feed_vae_low(pose_vae, condition, latent_z):

    output_shape = (-1, pose_vae.frame_size)

    vae_output = pose_vae.sample(latent_z, condition)
    vae_output = vae_output.view(output_shape)

    return vae_output

def feed_vae_up(pose_vae, condition, latent_z):

    output_shape = (-1, pose_vae.frame_size_rec)

    vae_output = pose_vae.sample(latent_z, condition)
    vae_output = vae_output.view(output_shape)

    return vae_output

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr, final_lr=0):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr - final_lr) * epoch / float(total_num_epochs)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def feed_vae(pose_vae, ground_truth, condition, future_weights, latent_z):
    condition = condition.flatten(start_dim=1, end_dim=2)
    #flattened_truth = ground_truth.flatten(start_dim=1, end_dim=2)
    output_shape = (-1, pose_vae.num_future_predictions, pose_vae.frame_size)

    vae_output = pose_vae.sample(latent_z, condition)
    vae_output = vae_output.view(output_shape)

    recon_loss = (vae_output - ground_truth).pow(2).mean(dim=(0, -1))
    recon_loss = recon_loss.mul(future_weights).sum()

    return vae_output, recon_loss

def feed_vae_eval(pose_vae, ground_truth, condition, latent_z):
    output_shape = (-1, pose_vae.frame_size)


    vae_output = pose_vae.sample(latent_z, condition)
    vae_output = vae_output.view(output_shape)

    recon_loss = (vae_output - ground_truth).pow(2).mean()

    return vae_output, recon_loss

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
    def __init__(self, args):
        self.start = time.time()
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
        ep = data["epoch"]
        ep_recon_loss = data["ep_recon_loss"]
        ep_z_mu_loss = data["ep_z_mu_loss"]
        ep_z_var_loss = data["ep_z_var_loss"]
        ep_none_loss = data['ep_none_loss']

        print(
            "{} | Recon: {:.4e} | z_mu: {:.4e} | z_var: {:.4e} | none: {:.4e}".format(
                self.time_since(ep), ep_recon_loss, ep_z_mu_loss, ep_z_var_loss, ep_none_loss
            ),
            flush=True,
        )

def cal_delta_facing(positions):
    right_v = positions[:,1,:] - positions[:,0,:]
    left_v = positions[:,5,:] - positions[:,0,:]
    facing = np.cross(right_v, left_v)
    facing_proj = facing[:,:2]
    if np.amin(facing_proj[:,0])>0:
        facing_proj_angles = np.arctan(facing_proj[:,1]/facing_proj[:,0])
    else:
        print('Error: Turning happened')
        pdb.set_trace()
    delta_facing = np.zeros((facing_proj_angles.shape[0],1))
    delta_facing[:,0] = facing_proj_angles - facing_proj_angles[0]

    return delta_facing

def cal_lbls(ipm_motions, id_f):
    delta_time = 1/60
    ln_v_thr = 0.15
    start_lbls = []
    for i in range(len(ipm_motions)):
        if i in id_f:
            start_lbls.append(0)
        else:
            pred_ipms = ipm_motions[i]
            pred_ipms_dot = (pred_ipms[1:] - pred_ipms[:-1]) / delta_time
            ln_v = torch.norm(pred_ipms_dot[:, :2], dim=-1)
            frame_num = pred_ipms_dot.shape[0]
            if ln_v.max() < ln_v_thr:
                start_lbls.append(-1)
            else:
                list_len_fore = len(start_lbls)
                for k in range(frame_num - 1):
                    if all([ln_v[k] >= ln_v_thr, ln_v[k + 1] >= ln_v_thr]):
                        if k < frame_num - 2:
                            start_lbls.append(k+5)
                        else:
                            start_lbls.append(-1)
                        break
                list_len_cur = len(start_lbls)
                if list_len_fore == list_len_cur:
                    start_lbls.append(-1)
    return  start_lbls

def cal_loss_mul_ipm(train_data, pred, pred_dot):
    loss_x = torch.abs(train_data[:, :, 0] - pred[:, :, 0]).mean()
    loss_y = torch.abs(train_data[:, :, 1] - pred[:, :, 1]).mean()
    loss_the = torch.abs(train_data[:, :, 2] - pred[:, :, 2]).mean()
    loss_phi = torch.abs(train_data[:, :, 3] - pred[:, :, 3]).mean()
    loss_x_dot = torch.abs(train_data[:, :, 4] - pred_dot[:, :, 0]).mean()
    loss_y_dot = torch.abs(train_data[:, :, 5] - pred_dot[:, :, 1]).mean()
    loss_the_dot = torch.abs(train_data[:, :, 6] - pred_dot[:, :, 2]).mean()
    loss_phi_dot = torch.abs(train_data[:, :, 7] - pred_dot[:, :, 3]).mean()
    return loss_x, loss_y, loss_the, loss_phi, loss_x_dot, loss_y_dot, loss_the_dot, loss_phi_dot

def cal_loss_sgl_ipm(train_data, pred, pred_dot):
    loss_x = torch.abs(train_data[:, 0] - pred[:, 0]).mean()
    loss_y = torch.abs(train_data[:, 1] - pred[:, 1]).mean()
    loss_the = torch.abs(train_data[:, 2] - pred[:, 2]).mean()
    loss_phi = torch.abs(train_data[:, 3] - pred[:, 3]).mean()
    loss_x_dot = torch.abs(train_data[:, 4] - pred_dot[:, 0]).mean()
    loss_y_dot = torch.abs(train_data[:, 5] - pred_dot[:, 1]).mean()
    loss_the_dot = torch.abs(train_data[:, 6] - pred_dot[:, 2]).mean()
    loss_phi_dot = torch.abs(pred_dot[:, 3]).mean()
    return loss_x, loss_y, loss_the, loss_phi, loss_x_dot, loss_y_dot, loss_the_dot, loss_phi_dot

def log_mul_ipm(logger, total_loss, total_loss_x, total_loss_y, total_loss_the, total_loss_phi, total_loss_x_dot, total_loss_y_dot,
                total_loss_the_dot, total_loss_phi_dot, total_num_mean, ep, split):
    total_loss /= total_num_mean
    total_loss_x /= total_num_mean
    total_loss_y /= total_num_mean
    total_loss_the /= total_num_mean
    total_loss_phi /= total_num_mean
    total_loss_x_dot /= total_num_mean
    total_loss_y_dot /= total_num_mean
    total_loss_the_dot /= total_num_mean
    total_loss_phi_dot /= total_num_mean

    logger.add_scalar(split + '/total_loss', total_loss, ep)
    logger.add_scalar(split + '/loss_x', total_loss_x, ep)
    logger.add_scalar(split + '/loss_y', total_loss_y, ep)
    logger.add_scalar(split + '/loss_the', total_loss_the, ep)
    logger.add_scalar(split + '/loss_phi', total_loss_phi, ep)
    logger.add_scalar(split + '/loss_x_dot', total_loss_x_dot, ep)
    logger.add_scalar(split + '/loss_y_dot', total_loss_y_dot, ep)
    logger.add_scalar(split + '/loss_the_dot', total_loss_the_dot, ep)
    logger.add_scalar(split + '/loss_phi_dot', total_loss_phi_dot, ep)
