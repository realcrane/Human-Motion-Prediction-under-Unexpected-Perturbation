from torch.utils.data import Dataset
import torch
import pickle
import os
import yaml
import numpy as np
from fairmotion.data import bvh

class IpmSDataset(Dataset):
    def __init__(self, split_name, args):
        #wfdur if_with_input_force_duration_frames_in_training
        super(IpmSDataset, self).__init__()
        self.prntDir = args.prntDir
        self.dataPath = args.prntDir + '/data/raw_data/indiv_data'
        self.sub_names = ['X05', 'X07', 'X08', 'X09']
        print(split_name)
        print('Sub_Names:', self.sub_names)
        self.split_name = split_name
        self.deltaT = args.deltaT
        self.wfdur = args.wfdur
        self.allMass = {'X05': 67.0, 'X07': 89.0, 'X08': 120.0, 'X09': 63.0}

        self.skel2ipm()
        self.load_forces()
        self.load_data()

    def load_data(self):

        ipmSet, massSet, rodLenSet, cartHSet, outFsSet = [], [], [], [], []

        if self.split_name == 'train':
            smps_lists_path = self.dataPath + '/train_list.yaml'
        elif self.split_name == 'eval':
            smps_lists_path = self.dataPath + '/eval_list.yaml'
        else:
            print('Split_name Error')
            raise
        with open(smps_lists_path, 'rb') as file:
            seqs_lists = yaml.safe_load(file)
        for sub_name in self.sub_names:
            ipm_sts = self.allIpmSet[sub_name]
            push = self.allPushSet[sub_name]
            index_list = seqs_lists[sub_name]
            for i in index_list:
                gt_ipm_st = torch.from_numpy(ipm_sts[i][:, :4])  # frames * 4
                gt_ipm_st_dot = torch.zeros_like(gt_ipm_st)
                gt_ipm_st_dot[1:] = (gt_ipm_st[1:] - gt_ipm_st[:-1]) / self.deltaT
                gt_ipm_st_dot[0] = gt_ipm_st_dot[1] * 0.95
                ipmData = torch.cat((gt_ipm_st, gt_ipm_st_dot), dim=-1)
                ipmSet.append(ipmData)
                massSet.append(self.allMass[sub_name])
                rodLens = torch.from_numpy(ipm_sts[i])[0, 4]
                rodLenSet.append(rodLens)
                cart_hs = torch.from_numpy(ipm_sts[i])[0, 5]
                cartHSet.append(cart_hs)
                outFsSet.append(push[i])
        self.ipmSet, self.massSet, self.rodLenSet, self.cartHSet, self.outFsSet\
            = ipmSet, massSet, rodLenSet, cartHSet, outFsSet
    def skel2ipm(self):
        # Map the skeleton data (T, J, 3) to ipm data (T, 8) (x,y theta,phi,rod_l, cart_h, hip_x, hip_y)
        print('Skeleton to IPM')
        ipmSetLoad = {}
        for sub_name in self.sub_names:
            print(sub_name)
            seqs_path = self.dataPath + '/' + sub_name + ''
            seqs_list = os.listdir(seqs_path)
            seqs_ipm_sts_list = []
            for seq in seqs_list:
                seq_bvh_path = seqs_path + '/' + seq
                motion = bvh.load(seq_bvh_path)
                positions = motion.positions(local=False)
                positions_scale = positions / 1000
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
            ipmSetLoad[sub_name] = seqs_ipm_sts_list
        self.allIpmSet = ipmSetLoad
    def load_forces(self):
        print('Load Forces')
        forcesPath = self.prntDir + '/data/raw_data/indiv_forces/forces.pkl'
        with open(forcesPath, 'rb') as f:
            allPushSet = pickle.load(f)
        self.allPushSet = allPushSet

    def __len__(self):
        return len(self.ipmSet)
    def __getitem__(self, index):
        item_dic = {}
        item_dic['ipmSts'] = self.ipmSet[index]
        item_dic['mass'] = self.massSet[index]
        item_dic['rodLen'] = self.rodLenSet[index]
        item_dic['cartH'] = self.cartHSet[index]
        item_dic['outFs'] = self.outFsSet[index]
        return item_dic

