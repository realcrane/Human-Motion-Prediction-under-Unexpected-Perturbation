import torch
import numpy as np
import torch.nn.functional as F
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
