import argparse
import datetime
import os
import pickle
import torch
def parse_args():
    now = str(datetime.datetime.now())[:19].replace(':', '-').replace(' ', '_')

    parser = argparse.ArgumentParser(description='Train an IPM Motion Model for Single-person')
    # Training
    parser.add_argument('--gpu_index', type=int, default=2, help='gpu')
    parser.add_argument('--logDir', default='logs/' + now)
    parser.add_argument('--lr', default=0.0003)
    parser.add_argument('--save_dir', default='/models/saved_models/')
    parser.add_argument('--expId', default='_2.pt')
    parser.add_argument('--num_eps', type=int, default=500)
    parser.add_argument('-norm_mode', default='zscore')
    parser.add_argument('--deltaT', default= 1/60)
    parser.add_argument('--rsm', type=bool, default=False)
    parser.add_argument('--bat_sz', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--lambdaPhiDot', default=0.3)
    parser.add_argument('--wfdur', default=True, help='if_with_input_force_duration_frames_in_training')

    # Model
    ### self-pd (PD control)
    parser.add_argument('--ks_x', default=torch.tensor([30, 4]))
    parser.add_argument('--ks_y', default=torch.tensor([30, 4]))
    parser.add_argument('--ks_theta', default=torch.tensor([1500, 200]))
    parser.add_argument('--ks_phi', default=torch.tensor([1500, 200]))
    ### self-nn
    parser.add_argument('--snInputSz', default=7)
    parser.add_argument('--snOutptSz', default=4)
    parser.add_argument('--snEmdSz', default=32)
    parser.add_argument('--snRnnSz', default=256)
    ### rod
    parser.add_argument('--rodInputSz', default=12)
    parser.add_argument('--rodOutptSz', default=1)
    parser.add_argument('--rodHdSz', default=[256, 128, 256])
    parser.add_argument('--minRod', default=0.871)
    parser.add_argument('--maxRod', default=1.075)

    #load
    parser.add_argument('--modelFsnnPath', default='/models/used_models/modelFsnn.pt')
    parser.add_argument('--modelRodPath', default='/models/used_models/modelRod.pt')




    args=parser.parse_args()
    return args
