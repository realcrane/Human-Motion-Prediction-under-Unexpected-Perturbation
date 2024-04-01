import copy
import torch
from fairmotion.data import bvh
import sys
import os
cur_dir = os.getcwd().replace('\\', '/')
prnt_dir = os.path.abspath(os.path.join(cur_dir, os.pardir)).replace('\\', '/')
sys.path.insert(0, prnt_dir)
from utils.utils import *
from blender.save_ipm_c3d import save_ipm_c3d

class IPM3D_sgl:
    def __init__(self, modelFsnn, modelRod, pdParams, mass, lPendulum, cartH, bFric, deltaT, numFrame,
                 iniSt, maxRod, minRod, ini_h_st, ini_c_st, iniStDot, outFs, device='cpu'):
        #We assume that the input force governs the motion during its duration D frames. This means that the the net force equals
        #the input force during the first D frames, and then equals f_self + friction after D frames.
        #As the input force is given, we assume that we know the ground truth of the IPM motion in the first D frames.
        #We just use the data including the IPM motion after the Dth frame to train the Differentiable IPM.

        self.modelFsnn = modelFsnn.to(device)
        self.modelRod = modelRod.to(device)
        self.pdParams = copy.deepcopy(pdParams)
        self.totMass = torch.tensor([mass]).to(device)
        self.M, self.m = self.totMass * 0.1, self.totMass * 0.9 # m_cart, m_pendulum
        self.l, self.cartH, self.b = lPendulum.to(device), cartH.to(device), bFric.to(device)
        self.g = torch.tensor(9.81).to(device)
        self.cur_step = 0
        self.deltaT = torch.tensor(deltaT).to(device)
        self.device = device
        self.obj_st = torch.zeros(4).to(device)
        self.maxRod, self.minRod = maxRod, minRod
        self.numFrame = numFrame
        self.stList, self.stDotList, self.rodList = [], [], []
        self.stList.append(iniSt.to(device))
        self.stDotList.append(iniStDot.to(device))
        self.rodList.append(self.l.unsqueeze(0))
        self.hSts = torch.zeros((numFrame, modelFsnn.cell.hidden_size)).double().to(device).detach()
        self.cSts = torch.zeros((numFrame, modelFsnn.cell.hidden_size)).double().to(device).detach()
        self.hStsList, self.cStsList = [], []
        self.hStsList.append(ini_h_st)
        self.cStsList.append(ini_c_st)
        self.outFs = outFs
        self.dur = outFs.shape[0]

    def step(self):
        # The calculation of the net force
        # self_f
        inputFSelfNN = torch.cat((self.stList[self.cur_step][2:], self.stDotList[self.cur_step], self.totMass)).double()
        fSelfNN, hSt, cSt = self.modelFsnn(inputFSelfNN, self.hStsList[self.cur_step], self.cStsList[self.cur_step])
        # pd_f Note the change of pd_params
        fSelfPd = self.PD_control()
        # friction
        friction = torch.cat((-self.b * self.stDotList[self.cur_step][:2], torch.zeros(2).to(self.device)))
        # net_force
        if self.cur_step < self.dur:
            netForce = self.outFs[self.cur_step]
        else:
            netForce = fSelfNN + fSelfPd + friction
        inputRod = torch.cat((inputFSelfNN[:-1], fSelfNN, self.totMass, self.rodList[self.cur_step]), dim=-1)
        deltaRod = self.modelRod(inputRod)
        self.rodList.append((self.rodList[self.cur_step] + deltaRod).clamp(self.minRod, self.maxRod))
        mat_M, mat_C, mat_G = calculate_mats(self.M, self.m, self.rodList[self.cur_step + 1], self.stList[self.cur_step][2],
                                             self.stList[self.cur_step][3], self.stDotList[self.cur_step][2],
                                             self.stDotList[self.cur_step][3], self.g, self.device)
        st_ddot = torch.linalg.inv(mat_M) @ (netForce.unsqueeze(1) - mat_C - mat_G)
        st_dot_new = self.stDotList[self.cur_step] + st_ddot.squeeze() * self.deltaT
        st_dot_new_clip = torch.zeros_like(st_dot_new).to(self.device)
        st_dot_new_clip[:2] = st_dot_new[:2]
        st_dot_new_clip[2] = torch.clip(st_dot_new[2], -3.9342, 3.8116)
        st_dot_new_clip[3] = torch.clip(st_dot_new[3], -4.4811, 6.7407)
        st_new = self.stList[self.cur_step] + st_dot_new_clip * self.deltaT
        st_new_clip = torch.cat((st_new[:2], torch.clip(st_new[2], -1.571, 1.571).unsqueeze(0),
                                 torch.clip(st_new[3], -1.571, 1.571).unsqueeze(0)))

        self.cur_step += 1
        self.hStsList.append(hSt.detach().data)
        self.cStsList.append(cSt.detach().data)
        self.stList.append(st_new_clip)
        self.stDotList.append(st_dot_new_clip)

    def output(self):
        return torch.stack(self.stList, dim=0), torch.stack(self.stDotList, dim=0)

    def PD_control(self):
        if self.cur_step == 0:
            pd_last_st = torch.cat((self.stDotList[self.cur_step][:2], self.stList[self.cur_step][2:]))
        else:
            pd_last_st = torch.cat((self.stDotList[self.cur_step - 1][:2], self.stList[self.cur_step - 1][2:]))
        pd_cur_st = torch.cat((self.stDotList[self.cur_step][:2], self.stList[self.cur_step][2:]))

        cur_err = self.obj_st - pd_cur_st
        las_err = self.obj_st - pd_last_st
        keys = list(self.pdParams.keys())
        outPd = torch.zeros(len(keys)).to(self.device)
        for i in range(len(keys)):
            outPd[i] = self.pdParams[keys[i]][0] * cur_err[i] + self.pdParams[keys[i]][1] * (cur_err[i] - las_err[i])
        return outPd
    def save(self, save_name, main_paths):
        ipmSts = torch.stack(self.stList, dim=0)
        ipmSts = ipmSts.detach().numpy()
        rodLen = self.l.detach().numpy()
        cartH = self.cartH.detach().numpy()
        save_ipm_c3d(save_name, main_paths, ipmSts, rodLen, cartH)