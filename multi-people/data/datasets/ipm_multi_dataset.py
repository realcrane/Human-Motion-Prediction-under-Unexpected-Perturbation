from torch.utils.data import Dataset
import torch
import pickle
import os
import yaml
import numpy as np
from fairmotion.data import bvh

class IpmMDataset(Dataset):
    def __init__(self, split_name, delta_t):
        super(IpmMDataset, self).__init__()
        cur_dir = os.getcwd().replace('\\', '/')
        self.prnt_dir = cur_dir
        self.data_set_path = self.prnt_dir + '/data/raw_data/group_data'
        self.gr_names = ['group_1', 'row_1', 'row_2', 'wait_2']
        #self.gr_names = ['group_1']
        print(split_name)
        print('Group_Names:', self.gr_names)
        self.split_name = split_name
        self.delta_t = delta_t
        self.skel2ipm()
        self.load_forces()
        self.load_data()

    def load_data(self):
        mass_id_set_path = self.prnt_dir + '/data/raw_data/group_mass_id_set.pkl'
        with open(mass_id_set_path, 'rb') as f:
            mass_id_set = pickle.load(f)
        mass_dic = {'row_1_05': torch.tensor([67.0]).double(),
                    'row_1_07': torch.tensor([89.0]).double(),
                    'row_1_08': torch.tensor([120.0]).double(),
                    'row_1_09': torch.tensor([63.0]).double(),
                    'row_2_09': torch.tensor([63.0]).double(),
                    'row_2_10': torch.tensor([68.0]).double(),
                    'row_2_11': torch.tensor([105.3]).double(),
                    'row_2_18': torch.tensor([81.0]).double(),
                    'row_2_19': torch.tensor([78.0]).double(),
                    'wait_2_10': torch.tensor([68.0]).double(),
                    'wait_2_11': torch.tensor([85.8]).double(),
                    'wait_2_16': torch.tensor([80.2]).double(),
                    'wait_2_18': torch.tensor([104.0]).double(),
                    'wait_2_19': torch.tensor([78.0]).double(),
                    'group_1_05': torch.tensor([67.0]).double(),
                    'group_1_07': torch.tensor([89.0]).double(),
                    'group_1_08': torch.tensor([120.0]).double(),
                    'group_1_09': torch.tensor([63.0]).double()}

        ipm_set, mass_set, rod_len_set, cart_h_set, out_fs_set = [], [], [], [], []

        if self.split_name == 'train':
            smps_lists_path = self.data_set_path + '/train_list.yaml'
        elif self.split_name == 'eval':
            smps_lists_path = self.data_set_path + '/eval_list.yaml'
        else:
            print('Split_name Error')
            raise
        with open(smps_lists_path, 'rb') as file:
            smps_lists = yaml.safe_load(file)
        for gr_name in self.gr_names:
            ipm_sts = self.all_ipm_set[gr_name]
            push = self.all_push_set[gr_name]
            duration = self.all_dur_set[gr_name]
            index_list = smps_lists[gr_name]
            for i in index_list:
                gt_ipm_st = torch.from_numpy(ipm_sts[i][:, :, :4])  # num_of_agents * frames * 4
                gt_ipm_st_dot = torch.zeros_like(gt_ipm_st)
                gt_ipm_st_dot[:, 1:] = (gt_ipm_st[:, 1:] - gt_ipm_st[:, :-1]) / self.delta_t
                gt_ipm_st_dot[:, 0] = gt_ipm_st_dot[:, 1] * 0.95
                train_data = torch.cat((gt_ipm_st, gt_ipm_st_dot), dim=-1)
                ipm_set.append(train_data)
                mass_id_list = mass_id_set[gr_name][i]
                mass_list = [mass_dic[key] for key in mass_id_list]
                mass_set.append(mass_list)
                rod_lens = torch.from_numpy(ipm_sts[i][:, 0, 4])
                rod_len_set.append(rod_lens)
                cart_hs = torch.from_numpy(ipm_sts[i][:, 0, 5])
                cart_h_set.append(cart_hs)
                out_fs_set.append(push[i][:duration[i]])
        self.ipm_set, self.mass_set, self.rod_len_set, self.cart_h_set, self.out_fs_set\
            = ipm_set, mass_set, rod_len_set, cart_h_set, out_fs_set
    def skel2ipm(self):
        # Map the skeleton data (T, J, 3) to ipm data (T, 8) (x,y theta,phi,rod_l, cart_h, hip_x, hip_y)
        print('Skeleton to IPM')
        ipm_set = {}
        for gr_name in self.gr_names:
            print(gr_name)
            smps_path = self.data_set_path + '/' + gr_name + ''
            smps_list = [str(idx) for idx in range(len(os.listdir(smps_path)))]
            smps_ipm_sts_list = []
            for smp in smps_list:
                smp_path = smps_path + '/' + smp
                seqs_list = os.listdir(smp_path)
                seqs_ipm_sts_list = []
                for seq_id in range(len(seqs_list)):
                    seq_bvh_path = smp_path + '/' + seqs_list[seq_id]
                    motion = bvh.load(seq_bvh_path)
                    positions = motion.positions(local=False)
                    if seq_id == 0:
                        translation_v = np.zeros((1, 1, 3))
                        # translation_v[0,0,:2] = positions[0,0,:2]
                    positions_tran = positions - translation_v

                    positions_scale = positions_tran / 1000
                    bvh_data_hip = positions_scale[:, 0, :]
                    bvh_data_cart = (positions_scale[:, 16, :] + positions_scale[:, 20, :]) / 2
                    rod_direction_norm = np.zeros((bvh_data_hip.shape[0], 1))
                    rod_direction_norm[:, 0] = np.linalg.norm((bvh_data_hip - bvh_data_cart), axis=-1)
                    rod_direction = (bvh_data_hip - bvh_data_cart) / rod_direction_norm
                    phi = np.arcsin(-rod_direction[:, 1:2])
                    theta = np.arcsin(rod_direction[:, 0:1] / np.cos(phi))

                    seq_ipm_sts = np.concatenate((bvh_data_cart[:, :-1], theta, phi, rod_direction_norm,
                                                 bvh_data_cart[:, -1:], bvh_data_hip[:, :-1]), axis=1)
                    seqs_ipm_sts_list.append(seq_ipm_sts)
                smp_ipm_sts = np.stack(seqs_ipm_sts_list, axis=0) #num_peds*frames*8
                smps_ipm_sts_list.append(smp_ipm_sts)
            ipm_set[gr_name] = smps_ipm_sts_list
        self.all_ipm_set = ipm_set
    def load_forces(self):
        print('Load Forces')
        forces_set_path = self.prnt_dir + '/data/raw_data/group_forces'
        dur_set, push_set = {}, {}

        for gr_name in self.gr_names:
            dur_path = forces_set_path + '/' + gr_name + '/duration.pkl'
            push_path = forces_set_path + '/' + gr_name + '/pushes.pkl'
            with open(dur_path, 'rb') as f:
                dur = pickle.load(f)
            with open(push_path, 'rb') as f:
                push = pickle.load(f)
            dur_set[gr_name] = dur
            push_set[gr_name] = push
        #correction:
        dur_set['row_1'][6], dur_set['row_2'][3] = 15, 20
        self.all_dur_set, self.all_push_set = dur_set, push_set

    def __len__(self):
        return len(self.ipm_set)
    def __getitem__(self, index):
        item_dic = {}
        item_dic['ipm_sts'] = self.ipm_set[index]
        item_dic['mass_list'] = self.mass_set[index]
        item_dic['rod_len_list'] = self.rod_len_set[index]
        item_dic['cart_h_list'] = self.cart_h_set[index]
        item_dic['out_fs'] = self.out_fs_set[index]
        return item_dic

