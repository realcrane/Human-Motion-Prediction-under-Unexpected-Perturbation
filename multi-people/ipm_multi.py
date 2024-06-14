import argparse
import datetime
import os
import pickle
import torch
def parse_args():
    cur_dir = os.getcwd().replace('\\', '/')
    prnt_dir = cur_dir
    inta_ag_dics_path = prnt_dir + '/utils/info/inta_ag_dics.pkl'
    with open(inta_ag_dics_path, 'rb') as file:
        inta_ag_dics = pickle.load(file)
    now = str(datetime.datetime.now())[:19].replace(':', '-').replace(' ', '_')

    parser = argparse.ArgumentParser(description='Train an IPM Motion Model for Multi-person')
    # Training
    parser.add_argument('--is_mp', action='store_true', help='if it is for multi-person')
    parser.add_argument('--gpu_index', type=int, default=0, help='gpu')
    parser.add_argument('--log_dir', default='runs_multi/' + now)
    parser.add_argument('--lr', default=0.0003)
    parser.add_argument('--save_dir', default=prnt_dir + '/saved_models/')
    parser.add_argument('--exp_id', default='_25.pt')
    parser.add_argument('--num_eps', type=int, default=100)
    parser.add_argument('-norm_mode', default='zscore')
    parser.add_argument('--delta_t', default= 1/60)
    parser.add_argument('--rsm', type=bool, default=False)
    parser.add_argument('--bat_sz', type=int, default=1)
    parser.add_argument('--cur_dir', default=cur_dir)
    parser.add_argument('--prnt_dir', default=prnt_dir)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--gr', default=True)
    parser.add_argument('--lambda_phi_dot', default=0.3)
    #parser.add_argument('--split_name', default='train')

    # Model
    ## Single-person
    ### self-pd (PD control)
    parser.add_argument('--ks_x', default=torch.tensor([30, 4]))
    parser.add_argument('--ks_y', default=torch.tensor([30, 4]))
    parser.add_argument('--ks_theta', default=torch.tensor([1500, 200]))
    parser.add_argument('--ks_phi', default=torch.tensor([1500, 200]))
    ### self-nn + f
    parser.add_argument('--self_nn_path', default=prnt_dir+'/models/used_models/self_f_sgl.pt')
    ### Rod Model
    parser.add_argument('--min_rod', default=0.871)
    parser.add_argument('--max_rod', default=1.075)
    parser.add_argument('--rod_model_path', default=prnt_dir+'/models/used_models/self_rod_sgl.pt')

    ## Multi-person
    parser.add_argument('--r_neigh', default=0.5)
    ### inta-bs
    #### inta-bs-xy
    parser.add_argument('--para_ln_V', default=150)
    #### inta-bs-ag
    parser.add_argument('--inta_the_dic', default = inta_ag_dics['inta_the_dic'])
    parser.add_argument('--inta_phi_dic', default = inta_ag_dics['inta_phi_dic'])
    parser.add_argument('--inta_ag_sc', default=100)
    parser.add_argument('--upr_thr', default=0.1, help='rad')
    #### inta-nn
    parser.add_argument('--inta_nn_path', default=prnt_dir+'/models/used_models/inta_f.pt')
    parser.add_argument('--inp_sz', type=int, default=10)
    parser.add_argument('--oup_sz', type=int, default=4)
    parser.add_argument('--h_sz', default=[512, 512])
    ### Standarization
    parser.add_argument('--inta_stdzn_inp_path', default=prnt_dir+'/utils/info/intera_stan_inputs.pkl')
    parser.add_argument('--max_inta_f', default=torch.tensor([[1587.3721, 1399.2835, 1323.8701, 832.9387]]))
    parser.add_argument('--min_inta_f', default=torch.tensor([[-2507.4078, -883.5459, -2106.6155, -1133.9879]]))
    parser.add_argument('--avg_inta_f', default=torch.tensor([[-46.7739, 6.6148, -43.1003, -8.8489]]))
    parser.add_argument('--std_inta_f', default=torch.tensor([[272.0406, 132.5160, 164.4270, 114.3698]]))

    # Data
    args=parser.parse_args()
    return args


# cur_dir = os.getcwd().replace('\\', '/')
# prnt_dir = os.path.abspath(os.path.join(cur_dir, os.pardir)).replace('\\', '/')
#tot_dir =  os.path.abspath(os.path.join(prnt_dir, os.pardir)).replace('\\', '/')
