from utils.utils import *
import torch.nn as nn
import random
import copy
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from scipy.spatial.transform import Rotation
import os
from fairmotion.data import bvh
import torch
import sys
cur_dir = os.getcwd().replace('\\', '/')
prnt_dir = os.path.abspath(os.path.join(cur_dir, os.pardir)).replace('\\', '/')
sys.path.insert(0, prnt_dir)


class IPM3D_mul:
    def __init__(self, model_f, model_rod, pd_params, mass, l_pendulum, cart_h, b_fric, delta_t, num_frames,
                 ini_st, max_rod, min_rod, ini_h_st, ini_c_st, ini_st_dot=None, out_fs=None, gr=False,  device='cpu',
                 if_pr = False, if_gen= False):
        # ini_force: frames * 4
        # Current Assumption: All people with a littler stronger average stability atfter self.sta_frame compared with the initial average stability
        #properties: stability (how difficult to be pushed to move); Stop (how quick to stop)
        # The invoke of self_f_nn and inta_f_nn should be considered further
        self.self_f_model = model_f.to(device)
        self.rod_model = model_rod.to(device)
        self.pd_params_ori = copy.deepcopy(pd_params)
        self.pd_params = copy.deepcopy(pd_params)
        self.total_m = torch.tensor([mass]).to(device)
        self.M = self.total_m * 0.1 # m_cart
        self.m = self.total_m * 0.9 # m_pendulum
        self.l = l_pendulum.to(device)
        self.cart_h = cart_h.to(device)
        self.b = b_fric.to(device)
        self.g = torch.tensor(9.81).to(device)
        self.line_v_thres = 0.01 #Measure the stablity
        self.angle_v_thres, self.angle_thres = 0.2, 0.05
        self.angle_mark_st, self.angle_sta_cou = 0, 0
        self.ln_mv_frame, self.ln_woe_frame = 25, 20
        if if_gen:
            self.ln_woe_pd_cof = 1.8
        else:
            self.ln_woe_pd_cof = 1
        self.angle_stop_step = 10
        self.cur_step = 0
        self.delta_t = torch.tensor(delta_t).to(device)
        self.gr, self.device = gr, device
        if out_fs == None:
            self.duration = 0
        elif torch.is_tensor(out_fs):
            self.out_fs = out_fs.to(device)
            self.duration = self.out_fs.shape[0]
        else:
            self.out_fs = torch.tensor(out_fs).to(device)
            self.duration = self.out_fs.shape[0]
        self.obj_st = torch.zeros(4).to(device)
        self.max_rod, self.min_rod = max_rod, min_rod
        self.num_frames = num_frames
        self.st = torch.zeros((num_frames, 4)).to(device)
        self.st_dot = torch.zeros((num_frames, 4)).to(device)
        self.st_list, self.st_dot_list = [], []
        self.rod = torch.zeros((num_frames, 1)).to(device)
        self.self_f = torch.zeros((num_frames, 4)).to(device)
        self.st[self.cur_step] = ini_st.to(device)
        self.st_list.append(ini_st.to(device))
        if ini_st_dot == None:
            self.st_dot[self.cur_step] = torch.zeros_like(ini_st).to(device)
            self.st_dot_list.append(torch.zeros_like(ini_st).to(device))
        else:
            self.st_dot[self.cur_step] = ini_st_dot.to(device)
            self.st_dot_list.append(ini_st_dot.to(device))
        self.rod[self.cur_step] = self.l
        self.h_sts = torch.zeros((num_frames, model_f.cell.hidden_size)).double().to(device).detach()
        self.c_sts = torch.zeros((num_frames, model_f.cell.hidden_size)).double().to(device).detach()
        self.h_sts_list, self.c_sts_list = [], []
        self.h_sts[self.cur_step] = ini_h_st
        self.c_sts[self.cur_step] = ini_c_st
        self.h_sts_list.append(ini_h_st)
        self.c_sts_list.append(ini_c_st)
        self.ln_mv_ct = 0
        self.mv_ct = 0
        self.wo_exf_ct = 0
        self.if_pr, self.if_gen = if_pr, if_gen
        self.v_thr = 0.05
        self.net_force = torch.zeros((num_frames, 4)).to(device)

    def step(self, intera_f=None):
        #Params adjustment for stopping
        line_v = torch.norm(self.st_dot[self.cur_step][:2].detach())
        v_abs = self.st_dot[self.cur_step].detach().abs().sum()
        if v_abs < self.v_thr:
            self.line_v_thres = 0.01
            self.mv_ct = 0
        else:
            self.mv_ct += 1
        if self.mv_ct > self.ln_mv_frame:
            if self.if_gen:
                #self.line_v_thres = 1
                self.line_v_thres = 0.05
            else:
                self.line_v_thres = 0.1 # 0.1 during evaluation; 1 during generation

        # The calculation of the net force
        # out_f
        if self.cur_step < self.duration:
            out_f = self.out_fs[self.cur_step]
        else:
            out_f = torch.zeros(4).to(self.device)
        # intera_f
        if intera_f == None:
            intera_f = torch.zeros_like(out_f).to(self.device)
        # self_f
        input_self_f = torch.cat((self.st[self.cur_step][2:].data, self.st_dot[self.cur_step].data, self.total_m)).double()
        self_f, h_st, c_st = self.self_f_model(input_self_f, self.h_sts_list[self.cur_step].data, self.c_sts_list[self.cur_step].data)
        if self.gr:
            if self.mv_ct > 30: #30 during evaluation
                if intera_f.abs().sum() == 0:
                    self_f_min, self_f_max = -50, 50
                else:
                    #self_f_min, self_f_max = -intera_f.detach().abs(), intera_f.detach().abs()
                    self_f_min, self_f_max = -intera_f.abs(), intera_f.abs()
                self_f = self_f.clamp(min=self_f_min, max=self_f_max)
            self_f = self_f.detach()
        self.self_f[self.cur_step] = self_f
        # pd_f Note the change of pd_params
        pd_f = self.PD_control()
        # friction
        friction = torch.cat((-self.b * self.st_dot[self.cur_step][:2].detach(), torch.zeros(2).to(self.device)))
        # net_force
        net_force = out_f + intera_f + self_f + pd_f + friction
        # if self.cur_step < self.duration:
        #     net_force = out_f + (intera_f + self_f + pd_f + friction) * 0.1
        #self.net_force[self.cur_step] = net_force
        # if self.if_pr:
        #     print('cur_step:', self.cur_step)
        #     print('cur_st:', self.st[self.cur_step][2:])
        #     print('cur_st_dot:', self.st_dot[self.cur_step])
        #     print('net_force:', net_force)
        #     print('self_f:', self_f)
        #     print('inta_f:', intera_f)
        #     print('mv_ct', self.mv_ct)
        #     print()
        #net_force = out_f + intera_f + self_f + friction

        inputs_rod = torch.cat((input_self_f[:-1], self_f, self.total_m, self.rod[self.cur_step]), dim=-1)
        delta_rod = self.rod_model(inputs_rod)
        self.rod[self.cur_step + 1] = (self.rod[self.cur_step] + delta_rod).clamp(self.min_rod, self.max_rod)
        mat_M, mat_C, mat_G = calculate_mats(self.M, self.m, self.rod[self.cur_step + 1].detach(), self.st[self.cur_step][2].detach(),
                                             self.st[self.cur_step][3].detach(), self.st_dot[self.cur_step][2].detach(),
                                             self.st_dot[self.cur_step][3].detach(), self.g, self.device)
        st_ddot = torch.linalg.inv(mat_M) @ (net_force.unsqueeze(1) - mat_C - mat_G)
        st_dot_new = self.st_dot[self.cur_step].detach() + st_ddot.squeeze() * self.delta_t
        st_dot_new_clip = torch.zeros_like(st_dot_new).to(self.device)
        if self.gr:
            st_dot_new_clip[0] = torch.clip(st_dot_new[0], -0.6360, 3.0901)
            st_dot_new_clip[1] = torch.clip(st_dot_new[1], -1.3357, 1.5026)
            st_dot_new_clip[2] = torch.clip(st_dot_new[2], -1.8478, 1.6746)
            st_dot_new_clip[3] = torch.clip(st_dot_new[3], -1.7310, 1.1985)
        else:
            st_dot_new_clip[:2] = st_dot_new[:2]
            st_dot_new_clip[2] = torch.clip(st_dot_new[2], -3.9342, 3.8116)
            st_dot_new_clip[3] = torch.clip(st_dot_new[3], -4.4811, 6.7407)
            # st_dot_new_clip[0] = torch.clip(st_dot_new[0], -0.6360, 3.0901)
            # st_dot_new_clip[1] = torch.clip(st_dot_new[1], -1.3357, 1.5026)
            # st_dot_new_clip[2] = torch.clip(st_dot_new[2], -1.8478, 1.6746)
            # st_dot_new_clip[3] = torch.clip(st_dot_new[3], -1.7310, 1.1985)
        st_new = self.st[self.cur_step].detach() + st_dot_new_clip * self.delta_t
        #st_new_clip = torch.zeros_like(st_new).to(self.device)
        st_new_clip = torch.cat((st_new[:2], torch.clip(st_new[2], -1.571, 1.571).unsqueeze(0),
                                 torch.clip(st_new[3], -1.571, 1.571).unsqueeze(0)))
        # st_new_clip[:2] = st_new[:2]
        # st_new_clip[2] = torch.clip(st_new[2], -1.571, 1.571)
        # st_new_clip[3] = torch.clip(st_new[3], -1.571, 1.571)

        # Params adjustment for stopping
        ex_f = out_f.abs().sum() + intera_f.abs().sum()
        if ex_f < 1:
            self.wo_exf_ct += 1
        else:
            self.wo_exf_ct = 0
        #correction
        if self.gr:
            st_new_clip, st_dot_new_clip = self.corr(ex_f, line_v, st_new_clip, st_dot_new_clip)
        else:
            if self.if_gen:
                st_new_clip, st_dot_new_clip = self.corr(ex_f, line_v, st_new_clip, st_dot_new_clip)
        self.cur_step += 1
        if self.gr:
            #self.h_sts[self.cur_step], self.c_sts[self.cur_step] = h_st.detach(), c_st.detach()
            self.h_sts_list.append(h_st.detach().data)
            self.c_sts_list.append(c_st.detach().data)
        else:
            self.h_sts_list.append(h_st.detach().data)
            self.c_sts_list.append(c_st.detach().data)
        self.st[self.cur_step] = st_new_clip.detach()
        self.st_dot[self.cur_step] = st_dot_new_clip.detach()
        self.st_list.append(st_new_clip)
        self.st_dot_list.append(st_dot_new_clip)

    def output(self):
        if self.gr:
            return self.st, self.st_dot, self.net_force
        else:
            return torch.stack(self.st_list, dim=0), torch.stack(self.st_dot_list, dim=0), self.net_force

    def save(self, save_name, main_paths):
        if torch.is_tensor(self.st):
            ipm_sts = self.st.detach().numpy()
        else:
            ipm_sts = self.st
        if torch.is_tensor(self.l):
            rod_len = self.l.detach().numpy()
        else:
            rod_len = self.l
        if torch.is_tensor(self.cart_h):
            cart_h = self.cart_h.detach().numpy()
        else:
            cart_h = self.cart_h
        save_ipm_c3d(save_name, main_paths, ipm_sts, rod_len, cart_h)

    def corr(self, ex_f, line_v, st_new_clip, st_dot_new_clip):
        if ex_f < 1 and line_v < self.line_v_thres:
            st_dot_new_clip[:2] = torch.zeros_like(st_dot_new_clip[:2]).to(self.device)
            st_new_clip[:2] = self.st[self.cur_step][:2]

        if ex_f < 1 and all([self.st[self.cur_step][2].abs() < self.angle_thres, self.st[self.cur_step][3].abs() < self.angle_thres,
                             self.st_dot[self.cur_step][2].abs() < self.angle_v_thres, self.st_dot[self.cur_step][3].abs() < self.angle_v_thres]):
            # state_dot_new_clipped[2:] = torch.zeros_like(state_dot_new[2:])
            if self.angle_sta_cou == 0:
                self.angle_mark_st = self.st[self.cur_step][2:]

            if self.angle_sta_cou < self.angle_stop_step:
                st_dot_new_clip[2:] = self.st_dot[self.cur_step][2:]
                st_new_clip[2:] = self.st[self.cur_step][2:] - self.angle_mark_st / self.angle_stop_step
                self.angle_sta_cou += 1
            else:
                st_dot_new_clip[2:] = torch.zeros(2).to(self.device)
                st_new_clip[2:] = torch.zeros(2).to(self.device)
        else:
            self.angle_sta_cou = 0

        return st_new_clip, st_dot_new_clip

    def PD_control(self):
        if self.wo_exf_ct > self.ln_woe_frame:
            self.pd_params['x'] = self.pd_params_ori['x'] * self.ln_woe_pd_cof  #1 during evaluation, 1.8 during generation
            self.pd_params['y'] = self.pd_params_ori['y'] * self.ln_woe_pd_cof
        else:
            self.pd_params['x'], self.pd_params['y'] = self.pd_params_ori['x'], self.pd_params_ori['y']
        if self.cur_step == 0:
            pd_last_st = torch.cat((self.st_dot[self.cur_step][:2], self.st[self.cur_step][2:]))
        else:
            pd_last_st = torch.cat((self.st_dot[self.cur_step - 1][:2], self.st[self.cur_step - 1][2:]))
        pd_cur_st = torch.cat((self.st_dot[self.cur_step][:2], self.st[self.cur_step][2:]))

        cur_err = self.obj_st - pd_cur_st
        las_err = self.obj_st - pd_last_st
        keys = list(self.pd_params.keys())
        out = torch.zeros(len(keys)).to(self.device)
        for i in range(len(keys)):
            out[i] = self.pd_params[keys[i]][0] * cur_err[i] + self.pd_params[keys[i]][1] * (cur_err[i] - las_err[i])
        return out

class IPM3D_sgl:
    def __init__(self, model_fsnn, model_rod, pd_params, mass, l_pendulum, cart_h, b_fric, delta_t, num_frames,
                 ini_st, max_rod, min_rod, ini_h_st, ini_c_st, ini_st_dot, out_fs,  device='cpu', if_gen= False):
        # ini_force: frames * 4
        # Current Assumption: All people with a littler stronger average stability atfter self.sta_frame compared with the initial average stability
        #properties: stability (how difficult to be pushed to move); Stop (how quick to stop)
        # The invoke of self_f_nn and inta_f_nn should be considered further
        self.model_fsnn = model_fsnn.to(device)
        self.model_rod = model_rod.to(device)
        self.pd_params_ori = copy.deepcopy(pd_params)
        self.pd_params = copy.deepcopy(pd_params)
        self.total_m = torch.tensor([mass]).to(device)
        self.M = self.total_m * 0.1 # m_cart
        self.m = self.total_m * 0.9 # m_pendulum
        self.l = l_pendulum.to(device)
        self.cart_h = cart_h.to(device)
        self.b = b_fric.to(device)
        self.g = torch.tensor(9.81).to(device)
        self.line_v_thres = 0.01 #Measure the stablity
        self.angle_v_thres, self.angle_thres = 0.2, 0.05
        self.angle_mark_st, self.angle_sta_cou = 0, 0
        self.ln_mv_frame, self.ln_woe_frame = 25, 20
        if if_gen:
            self.ln_woe_pd_cof = 1.8
        else:
            self.ln_woe_pd_cof = 1
        self.angle_stop_step = 10
        self.cur_step = 0
        self.delta_t = torch.tensor(delta_t).to(device)
        self.device = device
        if torch.is_tensor(out_fs):
            self.out_fs = out_fs.to(device)
            self.duration = self.out_fs.shape[0]
        else:
            self.out_fs = torch.tensor(out_fs).to(device)
            self.duration = self.out_fs.shape[0]
        self.obj_st = torch.zeros(4).to(device)
        self.max_rod, self.min_rod = max_rod, min_rod
        self.num_frames = num_frames
        self.st_list, self.st_dot_list, self.rod_list = [], [], []
        self.st_list.append(ini_st.to(device))
        self.st_dot_list.append(ini_st_dot.to(device))
        self.rod_list.append(self.l.unsqueeze(0))
        self.f_self_nn = torch.zeros((num_frames, 4)).to(device)
        self.h_sts = torch.zeros((num_frames, model_fsnn.cell.hidden_size)).double().to(device).detach()
        self.c_sts = torch.zeros((num_frames, model_fsnn.cell.hidden_size)).double().to(device).detach()
        self.h_sts_list, self.c_sts_list = [], []
        self.h_sts_list.append(ini_h_st)
        self.c_sts_list.append(ini_c_st)
        self.ln_mv_ct = 0
        self.mv_ct = 0
        self.wo_exf_ct = 0
        self.if_gen = if_gen
        self.v_thr = 0.05
        self.net_force = torch.zeros((num_frames, 4)).to(device)

    def step(self):
        #Params adjustment for stopping
        line_v = torch.norm(self.st_dot_list[self.cur_step][:2].detach())
        v_abs = self.st_dot_list[self.cur_step].detach().abs().sum()
        if v_abs < self.v_thr:
            self.line_v_thres = 0.01
            self.mv_ct = 0
        else:
            self.mv_ct += 1
        if self.mv_ct > self.ln_mv_frame:
            if self.if_gen:
                self.line_v_thres = 1
            else:
                self.line_v_thres = 0.1 # 0.1 during evaluation; 1 during generation

        # The calculation of the net force
        # out_f
        if self.cur_step < self.duration:
            out_f = self.out_fs[self.cur_step]
        else:
            out_f = torch.zeros(4).to(self.device)
        # self_f
        input_f_self_nn = torch.cat((self.st_list[self.cur_step][2:], self.st_dot_list[self.cur_step], self.total_m)).double()
        f_self_nn, h_st, c_st = self.model_fsnn(input_f_self_nn, self.h_sts_list[self.cur_step], self.c_sts_list[self.cur_step])
        #self.f_self_nn[self.cur_step] = f_self_nn
        # pd_f Note the change of pd_params
        f_self_pd = self.PD_control()
        # friction
        friction = torch.cat((-self.b * self.st_dot_list[self.cur_step][:2], torch.zeros(2).to(self.device)))
        # net_force
        net_force = out_f + f_self_nn + f_self_pd + friction
        # if self.cur_step < self.duration:
        #     net_force = out_f + (intera_f + self_f + pd_f + friction) * 0.1
        #self.net_force[self.cur_step] = net_force

        inputs_rod = torch.cat((input_f_self_nn[:-1], f_self_nn, self.total_m, self.rod_list[self.cur_step]), dim=-1)
        delta_rod = self.model_rod(inputs_rod)
        self.rod_list.append((self.rod_list[self.cur_step] + delta_rod).clamp(self.min_rod, self.max_rod))
        mat_M, mat_C, mat_G = calculate_mats(self.M, self.m, self.rod_list[self.cur_step + 1], self.st_list[self.cur_step][2],
                                             self.st_list[self.cur_step][3], self.st_dot_list[self.cur_step][2],
                                             self.st_dot_list[self.cur_step][3], self.g, self.device)
        st_ddot = torch.linalg.inv(mat_M) @ (net_force.unsqueeze(1) - mat_C - mat_G)
        st_dot_new = self.st_dot_list[self.cur_step] + st_ddot.squeeze() * self.delta_t
        st_dot_new_clip = torch.zeros_like(st_dot_new).to(self.device)

        st_dot_new_clip[:2] = st_dot_new[:2]
        st_dot_new_clip[2] = torch.clip(st_dot_new[2], -3.9342, 3.8116)
        st_dot_new_clip[3] = torch.clip(st_dot_new[3], -4.4811, 6.7407)
        # st_dot_new_clip[0] = torch.clip(st_dot_new[0], -0.6360, 3.0901)
        # st_dot_new_clip[1] = torch.clip(st_dot_new[1], -1.3357, 1.5026)
        # st_dot_new_clip[2] = torch.clip(st_dot_new[2], -1.8478, 1.6746)
        # st_dot_new_clip[3] = torch.clip(st_dot_new[3], -1.7310, 1.1985)
        st_new = self.st_list[self.cur_step] + st_dot_new_clip * self.delta_t
        st_new_clip = torch.cat((st_new[:2], torch.clip(st_new[2], -1.571, 1.571).unsqueeze(0),
                                 torch.clip(st_new[3], -1.571, 1.571).unsqueeze(0)))

        # Params adjustment for stopping
        ex_f = out_f.abs().sum()
        if ex_f < 1:
            self.wo_exf_ct += 1
        else:
            self.wo_exf_ct = 0
        #correction
        if self.if_gen:
            st_new_clip, st_dot_new_clip = self.corr(ex_f, line_v, st_new_clip, st_dot_new_clip)
        self.cur_step += 1

        self.h_sts_list.append(h_st.detach().data)
        self.c_sts_list.append(c_st.detach().data)
        self.st_list.append(st_new_clip)
        self.st_dot_list.append(st_dot_new_clip)

    def output(self):
        return torch.stack(self.st_list, dim=0), torch.stack(self.st_dot_list, dim=0), self.net_force

    def save(self, save_name, main_paths):
        if torch.is_tensor(self.st):
            ipm_sts = self.st.detach().numpy()
        else:
            ipm_sts = self.st
        if torch.is_tensor(self.l):
            rod_len = self.l.detach().numpy()
        else:
            rod_len = self.l
        if torch.is_tensor(self.cart_h):
            cart_h = self.cart_h.detach().numpy()
        else:
            cart_h = self.cart_h
        save_ipm_c3d(save_name, main_paths, ipm_sts, rod_len, cart_h)

    def corr(self, ex_f, line_v, st_new_clip, st_dot_new_clip):
        if ex_f < 1 and line_v < self.line_v_thres:
            st_dot_new_clip[:2] = torch.zeros_like(st_dot_new_clip[:2]).to(self.device)
            st_new_clip[:2] = self.st[self.cur_step][:2]

        if ex_f < 1 and all([self.st[self.cur_step][2].abs() < self.angle_thres, self.st[self.cur_step][3].abs() < self.angle_thres,
                             self.st_dot[self.cur_step][2].abs() < self.angle_v_thres, self.st_dot[self.cur_step][3].abs() < self.angle_v_thres]):
            # state_dot_new_clipped[2:] = torch.zeros_like(state_dot_new[2:])
            if self.angle_sta_cou == 0:
                self.angle_mark_st = self.st[self.cur_step][2:]

            if self.angle_sta_cou < self.angle_stop_step:
                st_dot_new_clip[2:] = self.st_dot[self.cur_step][2:]
                st_new_clip[2:] = self.st[self.cur_step][2:] - self.angle_mark_st / self.angle_stop_step
                self.angle_sta_cou += 1
            else:
                st_dot_new_clip[2:] = torch.zeros(2).to(self.device)
                st_new_clip[2:] = torch.zeros(2).to(self.device)
        else:
            self.angle_sta_cou = 0

        return st_new_clip, st_dot_new_clip

    def PD_control(self):
        if self.wo_exf_ct > self.ln_woe_frame:
            self.pd_params['x'] = self.pd_params_ori['x'] * self.ln_woe_pd_cof  #1 during evaluation, 1.8 during generation
            self.pd_params['y'] = self.pd_params_ori['y'] * self.ln_woe_pd_cof
        else:
            self.pd_params['x'], self.pd_params['y'] = self.pd_params_ori['x'], self.pd_params_ori['y']
        if self.cur_step == 0:
            pd_last_st = torch.cat((self.st_dot_list[self.cur_step][:2], self.st_list[self.cur_step][2:]))
        else:
            pd_last_st = torch.cat((self.st_dot_list[self.cur_step - 1][:2], self.st_list[self.cur_step - 1][2:]))
        pd_cur_st = torch.cat((self.st_dot_list[self.cur_step][:2], self.st_list[self.cur_step][2:]))

        cur_err = self.obj_st - pd_cur_st
        las_err = self.obj_st - pd_last_st
        keys = list(self.pd_params.keys())
        out = torch.zeros(len(keys)).to(self.device)
        for i in range(len(keys)):
            out[i] = self.pd_params[keys[i]][0] * cur_err[i] + self.pd_params[keys[i]][1] * (cur_err[i] - las_err[i])
        return out

class group:
    def __init__(self, intera_model, peds_dic, stan_info_inputs_intera, avg_intera_f, std_intera_f, max_intera_f, min_intera_f, para_ln_V, para_ln_sa,
                 delta_t, inta_the_dic, inta_phi_dic, inta_ag_sc, r_neigh, upr_thr, ln_v_thr=0.1, gt_st_and_dot = torch.tensor([0]), device = 'cpu'):
        self.peds = peds_dic
        self.intera_model = intera_model.to(device)
        self.cur_step = 0
        self.peds_id = list(peds_dic.keys())
        self.num_peds = len(self.peds_id)
        self.avg_intera = stan_info_inputs_intera['avg'].to(device).reshape(1, -1)
        self.std_intera = stan_info_inputs_intera['std'].to(device).reshape(1, -1)
        self.avg_intera_f = avg_intera_f.to(device)
        self.std_intera_f = std_intera_f.to(device)
        self.max_intera_f = max_intera_f.to(device)
        self.min_intera_f = min_intera_f.to(device)
        self.inta_f_bss = inta_f_bss(para_ln_V, para_ln_sa, delta_t, inta_the_dic, inta_phi_dic, inta_ag_sc, r_neigh, inta_upr_thr = upr_thr, device=device)
        self.gt_st_and_dot = gt_st_and_dot.to(device)
        self.ln_v_thr = ln_v_thr
        self.device = device
    def intera_force(self, r_neigh):
        cur_sts_list = []
        cur_sts_dot_list = []
        intera_fs = torch.zeros((self.num_peds, 4)).to(self.device)
        for key in self.peds_id:
            cur_st = self.peds[key].st[self.cur_step]
            cur_st_dot = self.peds[key].st_dot[self.cur_step]
            cur_sts_list.append(cur_st)
            cur_sts_dot_list.append(cur_st_dot)
        cur_sts = torch.stack(cur_sts_list, dim=0)
        cur_sts_dot = torch.stack(cur_sts_dot_list, dim=0)

        for i in range(self.num_peds):
            dis_neigh = torch.norm(cur_sts[i:i + 1, :2] - cur_sts[:, :2], dim=-1)
            dis_neigh[i] = r_neigh + 1
            dis_neigh_bool = dis_neigh < r_neigh
            cur_ln_v = torch.norm(cur_sts_dot[:,:2], dim=-1)
            ln_v_neigh_bool = cur_ln_v > self.ln_v_thr
            neigh_bool = torch.logical_and(dis_neigh_bool, ln_v_neigh_bool)
            neigh_st, neigh_st_dot = cur_sts[neigh_bool], cur_sts_dot[neigh_bool]
            num_neigh = neigh_st.shape[0]
            if num_neigh == 0:
                inta_f = torch.zeros(4).to(self.device)
            else:
                delta_st = cur_sts[i:i + 1, :] - neigh_st
                delta_st_dot = cur_sts_dot[i:i + 1, :] - neigh_st_dot
                main_agent = cur_sts[i].repeat(num_neigh, 1)
                inputs_intera = torch.cat((delta_st[:, :2], delta_st_dot[:, :2], main_agent[:, 2:3], neigh_st[:, 2:3],
                                           main_agent[:, 3:4], neigh_st[:, 3:4], delta_st_dot[:, 2:]), dim=-1)
                inputs_intera_stan = (inputs_intera - self.avg_intera) / self.std_intera
                inta_f_nn = (self.intera_model(inputs_intera_stan.double()) * self.std_intera_f + self.avg_intera_f).clamp(
                    min=self.min_intera_f, max=self.max_intera_f).sum(dim=0)
                inta_f_bss = self.inta_f_bss.cal(cur_sts[i], cur_sts_dot[i], neigh_st, neigh_st_dot)
                inta_f = inta_f_bss + inta_f_nn
                #inta_f = inta_f_nn
            intera_fs[i] = inta_f
        return intera_fs

    def step(self, intera_fs):
        for k in self.peds_id:
            self.peds[k].step(intera_fs[k])
        self.cur_step += 1

    def output(self):
        ipm_motions_list = []
        ipm_motions_dot_list = []
        net_force_list = []
        for i in range(self.num_peds):
            ipm_motion, ipm_motions_dot, net_force = self.peds[i].output()
            ipm_motions_list.append(ipm_motion)
            ipm_motions_dot_list.append(ipm_motions_dot)
            net_force_list.append(net_force)
        return torch.stack(ipm_motions_list, dim=0), torch.stack(ipm_motions_dot_list, dim=0), torch.stack(net_force_list, dim=0)
    def save(self, main_path):
        for k in self.peds_id:
            self.peds[k].save(str(k), main_path)

class inta_f_bss:
    def __init__(self, para_ln_V, para_ln_sa, delta_t, inta_the_dic, inta_phi_dic, inta_ag_sc, r_neigh, inta_upr_thr=0.05, delta=1e-3, device='cpu'):
        # The critical condition between fore and back during calculating the inter_f_bass on angles might need to be considered.
        self.para_ln_V = para_ln_V
        self.para_ln_sa = para_ln_sa
        self.delta_t = delta_t
        self.device = device
        self.delta = delta
        self.dx = torch.tensor([self.delta, 0.0]).to(device)  # 2
        self.dy = torch.tensor([0.0, self.delta]).to(device)  # 2
        self.inta_the_dic = inta_the_dic
        self.inta_phi_dic = inta_phi_dic
        self.inta_ag_sc = inta_ag_sc
        self.inta_upr_thr = inta_upr_thr
        self.r_neigh = r_neigh
        self.cof_v = 1
        self.min_cof_wv = 0.8
        self.max_cof_wv = 10
        self.cof_r_min = 0.5
        self.r_rel_min = 0.1


    def cal(self, main_st, main_st_dot, neighs_st, neighs_st_dot):
        #assumption: we don't condiser the situation where the agent in neigh moves away from the main agent with a high velocity.
        v_rel = torch.norm((neighs_st_dot[:, :2] - main_st_dot[:2]), dim=-1)
        cof_wv = (self.cof_v * v_rel).unsqueeze(1).clamp(min=self.min_cof_wv, max=self.max_cof_wv)
        r_vel = torch.norm((neighs_st[:,:2] - main_st[:2]), dim=-1)
        cof_wr = (self.cof_r_min + (self.r_neigh - r_vel) * ((1 - self.cof_r_min) / (self.r_neigh - self.r_rel_min))).unsqueeze(1).clamp(max = 1)
        cof_wvr = cof_wv * cof_wr

        inta_f_ln = self.inta_f_ln(main_st, main_st_dot, neighs_st, neighs_st_dot, cof_wvr)
        inta_f_ag = self.inta_f_ag(main_st, neighs_st, self.inta_upr_thr, cof_wvr)
        return torch.cat((inta_f_ln, inta_f_ag))

    def inta_f_ln(self, main_st, main_st_dot, neighs_st, neighs_st_dot, cof_wv):
        r_mj = main_st[:2] - neighs_st[:, : 2]
        v_jm = neighs_st_dot[:, :2] - main_st_dot[:2]
        V = self.pot_ln(r_mj, v_jm) #peds
        dvdx = (self.pot_ln(r_mj + self.dx, v_jm) - V) / self.delta #peds
        dvdy = (self.pot_ln(r_mj + self.dy, v_jm) - V) / self.delta #peds
        grad_r_ab = torch.stack((dvdx, dvdy), dim=-1)
        inta_f_ln = ((-grad_r_ab) * cof_wv).sum(dim=0)
        return inta_f_ln

    def pot_ln(self, r_mj, v_jm):
        b = 0.5 * ((torch.norm(r_mj, dim=-1) + torch.norm((r_mj - v_jm * self.delta_t), dim=-1)).pow(2) - (
                    torch.norm(v_jm, dim=-1) * self.delta_t).pow(2)).pow(0.5)
        V = self.para_ln_V * torch.exp(-b/self.para_ln_sa)
        return V

    def inta_f_ag(self, main_st, neighs_st, inta_upr_thr, cof_wv):
        inta_f_bss_the = torch.zeros(1).to(self.device)
        inta_f_bss_phi = torch.zeros(1).to(self.device)
        mx, my = main_st[0], main_st[1]
        m_the_lbl = self.pose_the_lbl(main_st[2], inta_upr_thr)
        m_phi_lbl = self.pose_phi_lbl(main_st[3], inta_upr_thr)
        num_neighs = neighs_st.shape[0]
        for i in range(num_neighs):
            neigh_the_lbl = self.pose_the_lbl(neighs_st[i, 2], inta_upr_thr)
            neigh_phi_lbl = self.pose_phi_lbl(neighs_st[i, 3], inta_upr_thr)
            loc_the_lbl = self.loc_the_lbl(mx, neighs_st[i, 0])
            loc_phi_lbl = self.loc_phi_lbl(my, neighs_st[i, 1])
            inta_f_bss_the += self.inta_the_dic[m_the_lbl][neigh_the_lbl][loc_the_lbl] * self.inta_ag_sc * cof_wv[i]
            inta_f_bss_phi += self.inta_phi_dic[m_phi_lbl][neigh_phi_lbl][loc_phi_lbl] * self.inta_ag_sc * cof_wv[i]
        return torch.cat((inta_f_bss_the, inta_f_bss_phi))

    def pose_the_lbl(self, angle, inta_upr_thr):
        if angle.abs() < inta_upr_thr:
            lbl = 0
        elif angle > 0:
            lbl = 1
        else:
            lbl = 2
        return lbl
    def pose_phi_lbl(self, angle, inta_upr_thr):
        if angle.abs() < inta_upr_thr:
            lbl = 0
        elif angle < 0:
            lbl = 1
        else:
            lbl = 2
        return lbl
    def loc_the_lbl(self, m_x, n_x):
        nm_x = n_x - m_x
        if nm_x > 0:
            return 'fore'
        else:
            return 'back'
    def loc_phi_lbl(self, m_y, n_y):
        nm_y = n_y - m_y
        if nm_y > 0:
            return 'fore'
        else:
            return 'back'

class Human:
    def __init__(self, cvae_low, ipm_nn_low, cvae_up, ipm_nn_up, ipm_motions, ini_poses, ini_orientation, low_body, up_body,
                 stan_inputs_low, stan_inputs_up, ipm_g, gr, device):

        #Assumption pose_ini_st_dot = 0
        self.cvae_low = cvae_low
        self.ipm_nn_low = ipm_nn_low
        self.cvae_up = cvae_up
        self.ipm_nn_up = ipm_nn_up
        self.ipm_motions = ipm_motions
        self.num_peds = ipm_motions.shape[0]
        self.num_frames = ipm_motions.shape[1] - 1
        self.ini_poses = ini_poses
        self.ini_poses_tran, self.tran_vs = tran_pose(ini_poses, self.num_peds, device)
        self.ini_orientation = ini_orientation
        self.ipm_fts_low, self.ipm_fts_up = ipm_st2fts(ipm_motions, ini_poses, stan_inputs_low, stan_inputs_up, gr, device)
        self.ipm_fts_dim_low = self.ipm_fts_low.shape[-1]
        self.ipm_fts_dim_up = self.ipm_fts_up.shape[-1]
        self.low_body = low_body
        self.up_body = up_body
        self.num_joints_low = len(low_body)
        self.num_joints_up = len(up_body)
        self.num_joints = self.num_joints_low + self.num_joints_up
        self.ft_len_low = 111
        self.ft_len_up = 198
        self.ft_len_vae_up = 171
        self.cur_step = 0
        self.low_body_fts = torch.zeros((self.num_peds, self.num_frames, self.ft_len_low)).to(device)
        self.up_body_fts = torch.zeros((self.num_peds, self.num_frames, self.ft_len_vae_up)).to(device)
        self.ipm_g = ipm_g
        self.gr = gr
        # self.low_body = torch.zeros((self.num_peds, self.num_frames, self.num_joints_low, 3))
        # self.up_body = torch.zeros((self.num_peds, self.num_frames, self.num_joints_up, 3))
        # self.full_body = torch.zeros((self.num_peds, self.num_frames, self.num_joints_low, 3))
        #
        # self.low_body[:, self.cur_step, :, :] = self.ini_poses[:, self.low_body, :]
        # self.up_body[:, self.cur_step, :, :] = self.ini_poses[:, self.up_body, :]
        # self.full_body[:, self.cur_step, :, :] = self.ini_poses
        self.low_body_fts[:, self.cur_step, :] = ft_cal_low(self.ini_poses_tran, self.ini_orientation, self.ipm_fts_dim_low, stan_inputs_low,
                                                        self.num_peds, self.ft_len_low, self.num_joints_low, self.low_body)
        self.up_body_fts[:, self.cur_step, :] = ft_cal_up(self.ini_poses_tran, self.ini_orientation, self.ipm_fts_dim_up, stan_inputs_up,
                                                        self.num_peds, self.ft_len_vae_up, self.num_joints_up, self.up_body)
    def step(self):
        if self.ipm_g:
            if self.gr:
                latent_z_low, latent_z_low_mu, _ = self.ipm_nn_low(self.ipm_fts_low[:, self.cur_step + 1], self.low_body_fts[:, self.cur_step])
            else:
                latent_z_low = self.ipm_nn_low(torch.cat((self.ipm_fts_low[:, self.cur_step + 1], self.low_body_fts[:, self.cur_step]), dim=-1))
        else:
            latent_z_low = torch.randn((self.num_peds, 64)).double()

        vae_output_low = feed_vae_low(self.cvae_low, self.low_body_fts[:, self.cur_step], latent_z_low)
        self.low_body_fts[:, self.cur_step + 1] = vae_output_low

        # latent_z_up = self.ipm_nn_up(torch.cat((self.ipm_fts_up[:, self.cur_step + 1], self.up_body_fts[:, self.cur_step],
        #                                         self.low_body_fts[:, self.cur_step + 1, 3 : 3 + 3 * self.num_joints_low]), dim=-1))
        if self.ipm_g:
            if self.gr:
                latent_z_up, latent_z_up_mu, _ = self.ipm_nn_up(self.ipm_fts_low[:, self.cur_step + 1],
                                             torch.cat((self.up_body_fts[:, self.cur_step], self.low_body_fts[:, self.cur_step + 1, 3 : 3 + 3 * self.num_joints_low]),
                                                       dim=-1))
            else:
                latent_z_up= self.ipm_nn_up(torch.cat((self.ipm_fts_low[:, self.cur_step + 1], self.up_body_fts[:, self.cur_step], self.low_body_fts[:, self.cur_step + 1, 3 : 3 + 3 * self.num_joints_low]),
                                                       dim=-1))
        else:
            latent_z_up = torch.randn((self.num_peds, 64)).double()

        vae_output_up = feed_vae_up(self.cvae_up, torch.cat((self.up_body_fts[:, self.cur_step],
                                                             self.low_body_fts[:, self.cur_step + 1, 3 : 3 + 3 * self.num_joints_low]), dim=-1),
                                    latent_z_up)
        self.cur_step += 1
        self.up_body_fts[:, self.cur_step] = vae_output_up
    def output(self):
        low_motions_tran = self.cvae_low.denormalize(self.low_body_fts)[:, :, 3 : 3 + 3 * self.num_joints_low]
        up_motions_tran = self.cvae_up.denormalize(self.up_body_fts, if_rec = True)[:, :, 3 : 3 + 3 * self.num_joints_up]
        low_motions = low_motions_tran.reshape(self.num_peds, self.num_frames, self.num_joints_low, 3) + self.tran_vs.unsqueeze(1)
        up_motions = up_motions_tran.reshape(self.num_peds, self.num_frames, self.num_joints_up, 3) + self.tran_vs.unsqueeze(1)
        up_adp2low = low_motions[:, :, 0:1, :] - up_motions[:, :, 0:1, :]
        up_motions_adp = up_motions + up_adp2low
        full_motions = torch.cat((up_motions_adp, low_motions[:, :, 1:]), dim=2)
        return low_motions, up_motions_adp, full_motions

    def save(self, save_name, main_paths, full_motions):
        if torch.is_tensor(full_motions):
            full_motions = full_motions.detach().numpy()
        save_pose_c3d(save_name, main_paths, full_motions)

class Human_low:
    def __init__(self, cvae_low, ipm_nn_low, ipm_motions, ini_poses, ini_orientation, low_body, stan_inputs_low, device):

        #Assumption pose_ini_st_dot = 0
        self.cvae_low = cvae_low
        self.ipm_nn_low = ipm_nn_low
        self.ipm_motions = ipm_motions
        self.num_peds = ipm_motions.shape[0]
        self.num_frames = ipm_motions.shape[1] - 1
        self.ini_poses = ini_poses
        self.ini_poses_tran, self.tran_vs = tran_pose(ini_poses, self.num_peds, device)
        self.ini_orientation = ini_orientation
        self.ipm_fts_low = ipm_st2fts_low(ipm_motions, ini_poses, stan_inputs_low, device)
        self.ipm_fts_dim_low = self.ipm_fts_low.shape[-1]
        self.ipm_fts_dim_up = self.ipm_fts_up.shape[-1]
        self.low_body = low_body
        self.num_joints_low = len(low_body)
        self.ft_len_low = 111
        self.cur_step = 0
        self.low_body_fts = torch.zeros((self.num_peds, self.num_frames, self.ft_len_low)).to(device)
        # self.low_body = torch.zeros((self.num_peds, self.num_frames, self.num_joints_low, 3))
        # self.up_body = torch.zeros((self.num_peds, self.num_frames, self.num_joints_up, 3))
        # self.full_body = torch.zeros((self.num_peds, self.num_frames, self.num_joints_low, 3))
        #
        # self.low_body[:, self.cur_step, :, :] = self.ini_poses[:, self.low_body, :]
        # self.up_body[:, self.cur_step, :, :] = self.ini_poses[:, self.up_body, :]
        # self.full_body[:, self.cur_step, :, :] = self.ini_poses
        self.low_body_fts[:, self.cur_step, :] = ft_cal_low(self.ini_poses_tran, self.ini_orientation, self.ipm_fts_dim_low, stan_inputs_low,
                                                        self.num_peds, self.ft_len_low, self.num_joints_low, self.low_body)
    def step(self):
        latent_z_low, _, _ = self.ipm_nn_low(self.ipm_fts_low[:, self.cur_step + 1], self.low_body_fts[:, self.cur_step])
        vae_output_low = feed_vae_low(self.cvae_low, self.low_body_fts[:, self.cur_step], latent_z_low)
        self.cur_step += 1
        self.low_body_fts[:, self.cur_step] = vae_output_low
    def output(self):
        low_motions_tran = self.cvae_low.denormalize(self.low_body_fts)[:, :, 3 : 3 + 3 * self.num_joints_low]
        low_motions = low_motions_tran.reshape(self.num_peds, self.num_frames, self.num_joints_low, 3) + self.tran_vs.unsqueeze(1)
        return low_motions

    def save(self, save_name, main_paths, low_motions):
        if torch.is_tensor(low_motions):
            low_motions = low_motions.detach().numpy()
        save_pose_c3d_low(save_name, main_paths, low_motions)

class skel_data:
    def __init__(self, data_main_path, subsets, tr_index_lists, ts_index_lists, lower_body, upper_body, delta_time, hipv_thr=0.1):
        #subsets: list of names of subsets
        self.main_path = data_main_path
        self.subsets = subsets
        self.tr_idx_list = tr_index_lists
        self.ts_idx_list = ts_index_lists
        self.lower_body = lower_body
        self.upper_body = upper_body
        self.delta_t, self.hipv_thr = delta_time, hipv_thr
    def skel_feats_and_dynlbls(self):
        self.tr_skel_feats_list = []
        self.ts_skel_feats_list = []
        self.tr_gt_dyn_lbls_list = []
        self.ts_gt_dyn_lbls_list = []
        num_subs = len(self.subsets)
        for l in range(num_subs):
            sub_path = self.main_path + self.subsets[l]
            grs_list = os.listdir(sub_path)
            for i_gr in range(len(grs_list)):
                seqs_path = sub_path + grs_list[i_gr]
                seqs_list = os.listdir(seqs_path)
                for j in range(len(seqs_list)):
                    seq_path = seqs_path + '/' + seqs_list[j]
                    motion = bvh.load(seq_path)
                    positions_ori = motion.positions(local=False)[:, self.lower_body]  # (frames, lower_joints, 3)
                    trans_v = np.zeros((1,1,3))
                    trans_v[0, 0, :2] = positions_ori[0, 0, :2]
                    positions = positions_ori - trans_v
                    delta_xy = positions[:, 0, :2] - positions[0:1, 0, :2]  # frames*2
                    delta_facing = cal_delta_facing(positions)  # frames*1
                    velocities = positions[1:] - positions[:-1]  # (frames-1, joints,3)
                    orientations = motion.rotations(local=False)[:-1, self.lower_body, :, :2].reshape(-1, len(self.lower_body), 6)  # (frames-1,9,6)
                    append_data = np.concatenate((delta_xy[:-1], delta_facing[:-1], positions[:-1].reshape((-1, 27)),
                                        velocities.reshape((-1, 27)), orientations.reshape((-1, 54))), axis=1)

                    positions_scale = positions / 1000
                    bvh_data_hip = positions_scale[:, 0, :]
                    hip_vel = np.linalg.norm(bvh_data_hip[1:, :1] - bvh_data_hip[:-1, :1], axis=-1) / self.delta_t
                    if j == 0:
                        label = 0
                    else:
                        if hip_vel.max() < self.hipv_thr:
                            label = -1
                        else:
                            for k in range(len(hip_vel)):
                                if hip_vel[k] >= self.hipv_thr:
                                    label = k
                                    break

                    if i_gr in self.tr_idx_list:
                        self.tr_skel_feats_list.append(append_data)
                        self.tr_gt_dyn_lbls_list.append(label)
                    else:
                        self.ts_skel_feats_list.append(append_data)
                        self.ts_gt_dyn_lbls_list.append(label)

    def ts_pred_dynlbls(self, ipmsts_list, ln_v_thr = 0.15, dis_thr=0.65, corr_frs=1):
        ipmsts = [gr_ipm[i] for gr_ipm in ipmsts_list for i in range(gr_ipm.shape[0])]
        self.ts_pred_dyn_lbls_list = []
        for gr_ipm in ipmsts[:-3]:
            gr_lbls_list = []
            for j in range(gr_ipm.shape[0]):
                if j == 0:
                    gr_lbls_list.append(0)
                else:
                    pred_ipms = gr_ipm[j]
                    pred_ipms_fore = gr_ipm[j - 1]
                    dis_ipms = torch.norm(pred_ipms - pred_ipms_fore, dim=-1)
                    pred_ipms_dot = (pred_ipms[1:] - pred_ipms[:-1]) / self.delta_t
                    ln_v = torch.norm(pred_ipms_dot[:, :2], dim=-1)
                    frame_num = pred_ipms_dot.shape[0]
                    if ln_v.max() < ln_v_thr:
                        gr_lbls_list.append(-1)
                    else:
                        list_len_fore = len(gr_lbls_list)
                        for k in range(frame_num - 1):
                            if all([ln_v[k] >= ln_v_thr, ln_v[k + 1] >= ln_v_thr, dis_ipms[k] < dis_thr]):
                                if k < frame_num - 2:
                                    gr_lbls_list.append(k)
                                else:
                                    gr_lbls_list.append(-1)
                                break
                        list_len_cur = len(gr_lbls_list)
                        if list_len_fore == list_len_cur:
                            gr_lbls_list.append(-1)
                    if gr_lbls_list[-1] > -1 and gr_lbls_list[-2] >= gr_lbls_list[-1]:
                        gr_lbls_list[-1] = gr_lbls_list[-2] + corr_frs

            self.ts_pred_dyn_lbls_list.extend(gr_lbls_list)

        for gr_ipm in ipmsts[-3:]:  #in the subset 'group_1'
            gr_lbls_list = []
            for j in range(gr_ipm.shape[0]):
                if j == 0:
                    gr_lbls_list.append(0)
                else:
                    pred_ipms = gr_ipm[j]
                    pred_ipms_dot = (pred_ipms[1:] - pred_ipms[:-1]) / self.delta_t
                    ln_v = torch.norm(pred_ipms_dot[:, :2], dim=-1)
                    frame_num = pred_ipms_dot.shape[0]

                    if ln_v.max() < ln_v_thr:
                        gr_lbls_list.append(-1)
                    else:
                        list_len_fore = len(gr_lbls_list)
                        for k in range(frame_num - 1):
                            if ln_v[k] >= ln_v_thr:
                                gr_lbls_list.append(k)
                                break
                        list_len_cur = len(gr_lbls_list)
                        if list_len_fore == list_len_cur:
                            gr_lbls_list.append(-1)
            if gr_lbls_list[-1] <= min(gr_lbls_list[-2], gr_lbls_list[-3]):
                gr_lbls_list[-1] = min(gr_lbls_list[-2], gr_lbls_list[-3]) + corr_frs
            self.ts_pred_dyn_lbls_list.extend(gr_lbls_list)

class train_ipm_guide_low:
    def __init__(self, args):
        #mocap_data: normalized; mocap_data_ori: raw_data
        self.cave_low, self.ipm_nn_low = args.cvae_low, args.ipm_nn_low
        self.optimizer = args.optimizer
        self.device = args.device
        self.num_con_frs = args.num_condition_frames
        self.num_fut_preds = args.num_future_predictions
        self.num_steps_per_rollout = args.num_steps_per_rollout
        self.ipm_ft_s, self.skel_ft_s = args.ipm_feature_size, args.skeleton_feature_size
        self.fut_ws = args.future_weights
        self.num_eps, self.smp_sched = args.num_eps, args.sample_schedule
        self.logger, self.logger_time = args.logger, args.logger_time
        self.mini_b_s, self.ini_lr, self.fnl_lr = args.mini_batch_size, args.initial_lr, args.final_lr
        self.end_idx, self.bvh_order = args.end_indices, args.bvh_order
        self.mocap_data_ori_pos, self.euler = args.mocap_data_ori_pos, args.euler
        self.mocap_data, self.sel_idx = args.mocap_data, args.selectable_indices
        shape = (self.mini_b_s, self.num_con_frs, self.skel_ft_s)
        self.history = torch.empty(shape).to(args.device)
        self.ts_history = torch.empty((1, self.skel_ft_s)).to(args.device)
        self.num_peds = len(self.end_idx)
    def BP(self, ep):
        self.ipm_nn_low.train()
        ep_recon_loss, ep_ipm_loss, ep_foot_loss, ep_bone_loss = 0, 0, 0, 0
        ep_recon_loss_phy_pos, ep_recon_loss_phy_rot = 0, 0
        ep_z_mu_loss, ep_z_var_loss = 0, 0
        ep_total_loss = 0
        sampler = BatchSampler(SubsetRandomSampler(self.sel_idx), self.mini_b_s, drop_last=True,)
        update_linear_schedule(self.optimizer, ep, self.num_eps, self.ini_lr, self.fnl_lr)

        for num_mini_batch, indices in enumerate(sampler):
            t_indices = torch.LongTensor(indices)
            # condition is from newest...oldest, i.e. (t-1, t-2, ... t-n)
            condition_range = (t_indices.repeat((self.num_con_frs, 1)).t()
                    + torch.arange(self.num_con_frs - 1, -1, -1).long())

            t_indices += self.num_con_frs
            self.history[:, : self.num_con_frs].copy_(self.mocap_data[condition_range][:, :, self.ipm_ft_s:])
            history_input_guide = self.mocap_data[condition_range].squeeze()

            for offset in range(self.num_steps_per_rollout):
                use_student = torch.rand(1) < self.smp_sched[ep]
                prediction_range = (t_indices.repeat((self.num_fut_preds, 1)).t()
                        + torch.arange(offset, offset + self.num_fut_preds).long())
                ground_truth = self.mocap_data[prediction_range][:, :, self.ipm_ft_s:]
                gt_pos = self.mocap_data_ori_pos[prediction_range].squeeze()
                gt_euler = self.euler[prediction_range].squeeze()
                condition = self.history[:, : self.num_con_frs]
                gt_z, gt_z_mu, gt_z_var = self.cave_low.encode(ground_truth[:, 0, :], condition[:, 0, :])
                latent_z, latent_z_mu, latent_z_var = self.ipm_nn_low(history_input_guide[:, :15], history_input_guide[:, 15:])

                vae_output, recon_loss = feed_vae(self.cave_low, ground_truth, condition, self.fut_ws, latent_z)
                z_mu_loss = (latent_z_mu - gt_z_mu).pow(2).mean()
                z_var_loss = (latent_z_var - gt_z_var).pow(2).mean()
                self.optimizer.zero_grad()
                total_loss = z_mu_loss + z_var_loss
                total_loss.backward()
                self.optimizer.step()

                self.history = self.history.roll(1, dims=1)
                next_frame = vae_output[:, 0] if use_student else ground_truth[:, 0]
                next_input_guide = torch.cat((self.mocap_data[prediction_range].squeeze()[:, : self.ipm_ft_s], next_frame), dim=-1)
                self.history[:, 0].copy_(next_frame.detach())
                history_input_guide.copy_(next_input_guide.detach())
                ep_total_loss += float(total_loss) / self.num_steps_per_rollout
                ep_recon_loss += float(recon_loss) / self.num_steps_per_rollout
                ep_z_mu_loss += float(z_mu_loss) / self.num_steps_per_rollout
                ep_z_var_loss += float(z_var_loss) / self.num_steps_per_rollout

                vae_output_dn = self.cave_low.denormalize(vae_output.detach()[:, 0])
                pred_pos = vae_output_dn[:, 3:30].reshape((-1, 9, 3))
                pred_rot = vae_output_dn[:, 57:].reshape((-1, 9, 6))
                pred_rm = from_6D_to_rm(pred_rot).cpu().detach().numpy()
                pred_euler = np.zeros((len(pred_rm), 9, 3))
                for joint in range(9):
                    r = Rotation.from_matrix(pred_rm[:, joint])
                    pred_euler[:, joint] = r.as_euler(self.bvh_order, degrees=True)
                pred_euler = torch.from_numpy(pred_euler)
                ep_recon_loss_phy_pos += torch.norm(gt_pos - pred_pos, dim=-1).mean() / self.num_steps_per_rollout
                ep_recon_loss_phy_rot += torch.abs(gt_euler - pred_euler).mean() / self.num_steps_per_rollout
        avg_ep_total_loss = ep_total_loss / num_mini_batch
        avg_ep_recon_loss = ep_recon_loss / num_mini_batch
        avg_ep_z_mu_loss = ep_z_mu_loss / num_mini_batch
        avg_ep_z_var_loss = ep_z_var_loss / num_mini_batch
        avg_ep_recon_loss_phy_pos = ep_recon_loss_phy_pos / num_mini_batch
        avg_ep_recon_loss_phy_rot = ep_recon_loss_phy_rot / num_mini_batch
        self.logger.add_scalar('train/1_total_loss', avg_ep_total_loss, ep)
        self.logger.add_scalar('train/1_recon_loss', avg_ep_recon_loss, ep)
        self.logger.add_scalar('train/1_z_mu_loss', avg_ep_z_mu_loss, ep)
        self.logger.add_scalar('train/1_z_var_loss', avg_ep_z_var_loss, ep)
        self.logger.add_scalar('train/1_recon_loss_pos', avg_ep_recon_loss_phy_pos / 1000, ep)
        self.logger.add_scalar('train/1_recon_loss_rot', avg_ep_recon_loss_phy_rot, ep)

        self.logger_time.log_stats({"epoch": ep, "ep_recon_loss": avg_ep_recon_loss,
                "ep_z_mu_loss": avg_ep_z_mu_loss, "ep_z_var_loss": avg_ep_z_var_loss,
                "ep_none_loss": 0})
    def cal_loss_2_ontrain(self, ep):
        self.ipm_nn_low.eval()
        with torch.no_grad():
            prediction_dn_list = []
            ep_total_loss_2, ep_recon_loss_2 = 0, 0
            ep_z_mu_loss_2, ep_z_var_loss_2, count_steps = 0, 0, 0
            for i in range(self.num_peds):
                if i == 0:
                    num_steps = self.end_idx[0] + 1
                    condition_range = torch.tensor([[0]])
                else:
                    num_steps = self.end_idx[i] - self.end_idx[i - 1]
                    condition_range = torch.tensor(self.end_idx[i - 1]).reshape(1, 1).long() + 1
                count_steps += num_steps - 1
                self.ts_history.copy_(self.mocap_data[condition_range][:, 0, self.ipm_ft_s:])
                history_input_guide = self.mocap_data[condition_range].reshape((1, -1))
                pred = np.zeros((num_steps, self.skel_ft_s))
                pred[0] = copy.deepcopy(self.ts_history.squeeze().detach().cpu().numpy())
                for step in range(1, num_steps):
                    pred_range = (condition_range + torch.arange(step, step + 1).long())
                    ground_truth = self.mocap_data[pred_range][:, 0, self.ipm_ft_s:]
                    gt_z, gt_z_mu, gt_z_var = self.cave_low.encode(ground_truth, self.ts_history)
                    latent_z, latent_z_mu, latent_z_var = self.ipm_nn_low(history_input_guide[:, :self.ipm_ft_s],
                                                                        history_input_guide[:, self.ipm_ft_s:])

                    next_frame_vae_output, recon_loss = feed_vae_eval(self.cave_low, ground_truth, self.ts_history, latent_z)
                    z_mu_loss = (latent_z_mu - gt_z_mu).pow(2).mean()
                    z_var_loss = (latent_z_var - gt_z_var).pow(2).mean()
                    total_loss = z_mu_loss + z_var_loss

                    next_input_guide = torch.cat((self.mocap_data[pred_range][:, 0, :self.ipm_ft_s], next_frame_vae_output), dim=-1)
                    self.ts_history.copy_(next_frame_vae_output.detach())
                    history_input_guide.copy_(next_input_guide.detach())
                    pred[step] = next_frame_vae_output.detach().cpu().numpy()
                    ep_total_loss_2 += float(total_loss)
                    ep_recon_loss_2 += float(recon_loss)
                    ep_z_mu_loss_2 += float(z_mu_loss)
                    ep_z_var_loss_2 += float(z_var_loss)

                prediction_dn = self.cave_low.denormalize(pred, if_np=1)
                prediction_dn_list.append(prediction_dn)
            prediction_dn_all = np.concatenate(prediction_dn_list, axis=0)
            prediction_dn_all_pos = prediction_dn_all[:, 3:30].reshape((-1, 9, 3))
            prediction_dn_all_rot = prediction_dn_all[:, 57:].reshape((-1, 9, 6))
            prediction_dn_all_rm = from_6D_to_rm(torch.from_numpy(prediction_dn_all_rot)).detach().cpu().numpy()
            prediction_dn_all_euler = np.zeros((len(prediction_dn_all), 9, 3))
            for joint in range(9):
                r = Rotation.from_matrix(prediction_dn_all_rm[:, joint])
                prediction_dn_all_euler[:, joint] = r.as_euler(self.bvh_order, degrees=True)
            avg_ep_recon_loss_phy_pos_2 = np.linalg.norm(
                self.mocap_data_ori_pos.detach().cpu().numpy() - prediction_dn_all_pos, axis=-1).mean()
            avg_ep_recon_loss_phy_rot_2 = np.abs(self.euler.detach().cpu().numpy() - prediction_dn_all_euler).mean()

            avg_ep_total_loss_2 = ep_total_loss_2 / count_steps
            avg_ep_recon_loss_2 = ep_recon_loss_2 / count_steps
            avg_ep_z_mu_loss_2 = ep_z_mu_loss_2 / count_steps
            avg_ep_z_var_loss_2 = ep_z_var_loss_2 / count_steps
            self.logger.add_scalar('train/2_total_loss', avg_ep_total_loss_2, ep)
            self.logger.add_scalar('train/2_recon_loss', avg_ep_recon_loss_2, ep)
            self.logger.add_scalar('train/2_z_mu_loss', avg_ep_z_mu_loss_2, ep)
            self.logger.add_scalar('train/2_z_var_loss', avg_ep_z_var_loss_2, ep)
            self.logger.add_scalar('train/2_recon_loss_pos', avg_ep_recon_loss_phy_pos_2 / 1000, ep)
            self.logger.add_scalar('train/2_recon_loss_rot', avg_ep_recon_loss_phy_rot_2, ep)

class eval_ipm_guide_low:
    def __init__(self, args):
        self.cave_low, self.ipm_nn_low = args.cvae_low, args.ipm_nn_low
        self.device = args.device
        self.num_con_frs = args.num_condition_frames
        self.num_fut_preds = args.num_future_predictions
        self.ipm_ft_s, self.skel_ft_s = args.ipm_feature_size, args.skeleton_feature_size
        self.fut_ws = args.future_weights
        self.logger= args.logger
        self.ts_end_idx, self.bvh_order = args.ts_end_indices, args.bvh_order
        self.ts_mocap_data_ori_pos, self.ts_euler = args.ts_mocap_data_ori_pos, args.ts_euler
        self.ts_mocap_data = args.ts_mocap_data
        self.ts_history = torch.empty((1, self.skel_ft_s)).to(args.device)
        self.ts_num_peds = len(self.ts_end_idx)
    def eval(self):
        ts_ep_total_loss_2 = 0
        ts_ep_recon_loss_2 = 0
        ts_ep_z_mu_loss_2 = 0
        ts_ep_z_var_loss_2 = 0
        ts_count_steps = 0
        ts_pred_list = []
        ts_pred_dn_list = []
        for i in range(self.ts_num_peds):
            if i == 0:
                ts_num_steps = self.ts_end_idx[0] + 1
                ts_cond_range = torch.tensor([[0]])
            else:
                ts_num_steps = self.ts_end_idx[i] - self.ts_end_idx[i - 1]
                ts_cond_range = torch.tensor(self.ts_end_idx[i - 1]).reshape(1, 1).long() + 1
            ts_count_steps += ts_num_steps - 1
            self.ts_history.copy_(self.ts_mocap_data[ts_cond_range][:, 0, self.ipm_ft_s:])
            ts_history_input_guide = self.ts_mocap_data[ts_cond_range].reshape((1, -1))
            ts_pred = torch.zeros((ts_num_steps, self.skel_ft_s)).to(self.device)
            ts_pred[0] = copy.deepcopy(self.ts_history.squeeze())
            for step in range(1, ts_num_steps):
                ts_pred_range = (ts_cond_range + torch.arange(step, step + 1).long())
                ts_ground_truth = self.ts_mocap_data[ts_pred_range][:, 0, self.ipm_ft_s:]
                ts_gt_z, ts_gt_z_mu, ts_gt_z_var = self.cave_low.encode(ts_ground_truth, self.ts_history)
                ts_latent_z, ts_latent_z_mu, ts_latent_z_var = self.ipm_nn_low(ts_history_input_guide[:, :self.ipm_ft_s],
                                                                                   ts_history_input_guide[:, self.ipm_ft_s:])

                ts_next_frame_vae_output, ts_recon_loss = \
                    feed_vae_eval(self.cave_low, ts_ground_truth, self.ts_history, ts_latent_z)

                ts_z_mu_loss = (ts_latent_z_mu - ts_gt_z_mu).pow(2).mean()
                ts_z_var_loss = (ts_latent_z_var - ts_gt_z_var).pow(2).mean()
                ts_total_loss = ts_z_mu_loss + ts_z_var_loss

                ts_next_input_guide = torch.cat(
                    (self.ts_mocap_data[ts_pred_range][:, 0, :self.ipm_ft_s], ts_next_frame_vae_output), dim=-1)

                self.ts_history.copy_(ts_next_frame_vae_output.detach())
                ts_history_input_guide.copy_(ts_next_input_guide.detach())
                ts_pred[step] = ts_next_frame_vae_output
                ts_ep_total_loss_2 += float(ts_total_loss)
                ts_ep_recon_loss_2 += float(ts_recon_loss)
                ts_ep_z_mu_loss_2 += float(ts_z_mu_loss)
                ts_ep_z_var_loss_2 += float(ts_z_var_loss)

            ts_pred_dn = self.cave_low.denormalize(ts_pred)
            ts_pred_list.append(ts_pred)
            ts_pred_dn_list.append(ts_pred_dn)
        ts_pred_dn_all = torch.cat(ts_pred_dn_list, dim=0)
        ts_pred_dn_all_pos = ts_pred_dn_all[:, 3:30].reshape((-1, 9, 3))
        ts_pred_dn_all_rot = ts_pred_dn_all[:, 57:].reshape((-1, 9, 6))
        ts_pred_dn_all_rm = from_6D_to_rm(ts_pred_dn_all_rot).cpu().detach().numpy()
        ts_pred_dn_all_euler = np.zeros((len(ts_pred_dn_all), 9, 3))
        for joint in range(9):
            r = Rotation.from_matrix(ts_pred_dn_all_rm[:, joint])
            ts_pred_dn_all_euler[:, joint] = r.as_euler(self.bvh_order, degrees=True)
        ts_pred_dn_all_euler = torch.from_numpy(ts_pred_dn_all_euler)
        ts_avg_ep_recon_loss_phy_pos_2 = torch.norm(self.ts_mocap_data_ori_pos - ts_pred_dn_all_pos, dim=-1).mean()
        ts_avg_ep_recon_loss_phy_rot_2 = torch.abs(self.ts_euler - ts_pred_dn_all_euler).mean()

        ts_avg_ep_total_loss_2 = ts_ep_total_loss_2 / ts_count_steps
        ts_avg_ep_recon_loss_2 = ts_ep_recon_loss_2 / ts_count_steps
        ts_avg_ep_z_mu_loss_2 = ts_ep_z_mu_loss_2 / ts_count_steps
        ts_avg_ep_z_var_loss_2 = ts_ep_z_var_loss_2 / ts_count_steps
        return ts_avg_ep_total_loss_2, ts_avg_ep_recon_loss_2, ts_avg_ep_z_mu_loss_2, ts_avg_ep_z_var_loss_2, ts_avg_ep_recon_loss_phy_pos_2, ts_avg_ep_recon_loss_phy_rot_2




























