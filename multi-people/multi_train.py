from cfgs.ipm_multi import parse_args
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pickle
import sys
import torch.optim as optim
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def train():
    inta_nn.train()
    total_loss, total_num_mean = 0, 0
    total_loss_x, total_loss_y, total_loss_the, total_loss_phi = 0, 0, 0, 0
    total_loss_x_dot, total_loss_y_dot, total_loss_the_dot, total_loss_phi_dot = 0, 0, 0, 0
    for batch in trainLoader:
        train_data = batch['ipm_sts'][0].to(device)
        mass = batch['mass_list']
        rod_len = batch['rod_len_list'][0].to(device)
        cart_h = batch['cart_h_list'][0].to(device)
        out_forces = batch['out_fs'][0].to(device)
        num_peds = train_data.shape[0]
        num_frames = train_data.shape[1]
        ini_h_st = torch.zeros(self_nn.cell.hidden_size).double().to(args.device)
        ini_c_st = torch.zeros(self_nn.cell.hidden_size).double().to(args.device)

        peds_dic = {}
        for i in range(num_peds):
            if i == 0:
                peds_dic[i] = IPM3D_mul(self_nn, model_to_rod, pd_params, mass[i].to(device), rod_len[i], cart_h[i], miu_ground,
                                    args.delta_t,
                                    num_frames, train_data[i, 0, :4], args.max_rod, args.min_rod, ini_h_st, ini_c_st,
                                    train_data[i, 0, 4:], out_forces, args.gr, args.device)
            else:
                peds_dic[i] = IPM3D_mul(self_nn, model_to_rod, pd_params, mass[i].to(device), rod_len[i], cart_h[i], miu_ground,
                                    args.delta_t,
                                    num_frames, train_data[i, 0, :4], args.max_rod, args.min_rod, ini_h_st, ini_c_st,
                                    train_data[i, 0, 4:], gr=args.gr, device=args.device)
        gr = group(inta_nn, peds_dic, inta_stazn_inp, args.avg_inta_f, args.std_inta_f, args.max_inta_f,
                   args.min_inta_f,
                   args.para_ln_V, args.r_neigh, args.delta_t, args.inta_the_dic, args.inta_phi_dic, args.inta_ag_sc,
                   args.r_neigh, args.upr_thr, gt_st_and_dot=train_data, device=args.device)

        for _ in range(1, gr.peds[0].num_frames):
            intera_fs = gr.intera_force(args.r_neigh)
            gr.step(intera_fs)
        pred, pred_dot, _ = gr.output()
        loss_x, loss_y, loss_the, loss_phi, loss_x_dot, loss_y_dot, loss_the_dot, loss_phi_dot = cal_loss_mul_ipm(
            train_data, pred, pred_dot)
        num_mean = gr.num_peds * gr.peds[0].num_frames
        optimizer.zero_grad()
        loss = loss_x + loss_y + loss_the + loss_phi + loss_x_dot + loss_y_dot + loss_the_dot + args.lambda_phi_dot * loss_phi_dot
        loss.requires_grad_(True)
        loss.backward()
        nn.utils.clip_grad_norm_(inta_nn.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * num_mean
        total_loss_x += loss_x.item() * num_mean
        total_loss_y += loss_y.item() * num_mean
        total_loss_the += loss_the.item() * num_mean
        total_loss_phi += loss_phi.item() * num_mean
        total_loss_x_dot += loss_x_dot.item() * num_mean
        total_loss_y_dot += loss_y_dot.item() * num_mean
        total_loss_the_dot += loss_the_dot.item() * num_mean
        total_loss_phi_dot += loss_phi_dot.item() * num_mean
        total_num_mean += num_mean
    log_mul_ipm(logger, total_loss, total_loss_x, total_loss_y, total_loss_the, total_loss_phi, total_loss_x_dot,
                total_loss_y_dot,
                total_loss_the_dot, total_loss_phi_dot, total_num_mean, ep, 'train')
    total_loss /= total_num_mean
    return total_loss

def eval():
    inta_nn.eval()
    total_loss, total_num_mean = 0, 0
    total_loss_x, total_loss_y, total_loss_the, total_loss_phi = 0, 0, 0, 0
    total_loss_x_dot, total_loss_y_dot, total_loss_the_dot, total_loss_phi_dot = 0, 0, 0, 0
    for batch in evalLoader:
        eval_data = batch['ipm_sts'][0].to(device)
        mass = batch['mass_list']
        rod_len = batch['rod_len_list'][0].to(device)
        cart_h = batch['cart_h_list'][0].to(device)
        out_forces = batch['out_fs'][0].to(device)
        num_peds = eval_data.shape[0]
        num_frames = eval_data.shape[1]
        ini_h_st = torch.zeros(self_nn.cell.hidden_size).double().to(args.device)
        ini_c_st = torch.zeros(self_nn.cell.hidden_size).double().to(args.device)
        peds_dic = {}
        for i in range(num_peds):
            if i == 0:
                peds_dic[i] = IPM3D_mul(self_nn, model_to_rod, pd_params, mass[i].to(device), rod_len[i], cart_h[i], miu_ground,
                                    args.delta_t,
                                    num_frames, eval_data[i, 0, :4], args.max_rod, args.min_rod, ini_h_st, ini_c_st,
                                    eval_data[i, 0, 4:], out_forces, args.gr, args.device)
            else:
                peds_dic[i] = IPM3D_mul(self_nn, model_to_rod, pd_params, mass[i].to(device), rod_len[i], cart_h[i], miu_ground,
                                    args.delta_t,
                                    num_frames, eval_data[i, 0, :4], args.max_rod, args.min_rod, ini_h_st, ini_c_st,
                                    eval_data[i, 0, 4:], gr=args.gr, device=args.device)
        gr = group(inta_nn, peds_dic, inta_stazn_inp, args.avg_inta_f, args.std_inta_f, args.max_inta_f,
                   args.min_inta_f,
                   args.para_ln_V, args.r_neigh, args.delta_t, args.inta_the_dic, args.inta_phi_dic, args.inta_ag_sc,
                   args.r_neigh, args.upr_thr, gt_st_and_dot=eval_data, device=args.device)

        for _ in range(1, gr.peds[0].num_frames):
            intera_fs = gr.intera_force(args.r_neigh)
            gr.step(intera_fs)
        pred, pred_dot, _ = gr.output()
        loss_x, loss_y, loss_the, loss_phi, loss_x_dot, loss_y_dot, loss_the_dot, loss_phi_dot = cal_loss_mul_ipm(
            eval_data, pred, pred_dot)
        num_mean = gr.num_peds * gr.peds[0].num_frames
        loss = loss_x + loss_y + loss_the + loss_phi + loss_x_dot + loss_y_dot + loss_the_dot + args.lambda_phi_dot * loss_phi_dot
        total_loss += loss.item() * num_mean
        total_loss_x += loss_x.item() * num_mean
        total_loss_y += loss_y.item() * num_mean
        total_loss_the += loss_the.item() * num_mean
        total_loss_phi += loss_phi.item() * num_mean
        total_loss_x_dot += loss_x_dot.item() * num_mean
        total_loss_y_dot += loss_y_dot.item() * num_mean
        total_loss_the_dot += loss_the_dot.item() * num_mean
        total_loss_phi_dot += loss_phi_dot.item() * num_mean
        total_num_mean += num_mean
    log_mul_ipm(logger, total_loss, total_loss_x, total_loss_y, total_loss_the, total_loss_phi, total_loss_x_dot,
                total_loss_y_dot,
                total_loss_the_dot, total_loss_phi_dot, total_num_mean, ep, 'eval')
    total_loss /= total_num_mean
    return total_loss

args = parse_args()
sys.path.insert(0, args.prnt_dir)
from models.models import *
from data.datasets.ipm_multi_dataset import IpmMDataset
from utils.classes import *
from utils.utils import cal_loss_mul_ipm, log_mul_ipm
logger = SummaryWriter(args.log_dir)
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
print(device)
args.device = device

pd_params = {'x': args.ks_x.to(device), 'y': args.ks_y.to(device), 'theta': args.ks_theta.to(device), 'phi': args.ks_phi.to(device)}
with open(args.inta_stdzn_inp_path, 'rb') as f:
    inta_stazn_inp = pickle.load(f)

trainSet = IpmMDataset('train', args.delta_t)
evalSet = IpmMDataset('eval', args.delta_t)
trainLoader = DataLoader(trainSet, batch_size=args.bat_sz, num_workers=args.num_workers,
                        drop_last=False, shuffle=args.shuffle)
evalLoader = DataLoader(evalSet, batch_size=args.bat_sz, num_workers=args.num_workers,
                        drop_last=False, shuffle=False)

#Load individual-level IPM models
self_nn = LSTM_force_cell()
model_to_rod = NN_to_rod()
self_nn = self_nn.double().to(args.device)
model_to_rod = model_to_rod.double().to(args.device)
ckpt_self_nn = torch.load(args.self_nn_path, map_location=args.device)
self_nn.load_state_dict(ckpt_self_nn['model_state_dict'])
ckpt_rod = torch.load(args.rod_model_path, map_location=args.device)
model_to_rod.load_state_dict(ckpt_rod['model_state_dict'])
miu_ground = ckpt_self_nn['physics']
miu_ground.requires_grad = False
self_nn.eval()
model_to_rod.eval()
inta_nn = MLP(args.inp_sz, args.oup_sz, args.h_sz).double().to(args.device)
if args.rsm:
    ckpt_inta_nn = torch.load(args.inta_nn_path, map_location=args.device)
    inta_nn.load_state_dict(ckpt_inta_nn['model_state_dict'])
inta_nn.train()
optimizer = optim.Adam(inta_nn.parameters(), lr=args.lr)
best_loss_trn = 10e5
best_loss_ts = 10e5
best_ep_trn = 0
best_ep_ts = 0

for ep in range(args.num_eps):
    train_loss = train()
    if train_loss < best_loss_trn:
        best_loss_trn = train_loss
        best_ep_trn = ep
        print('saving models for training')
        torch.save({'model_state_dict': inta_nn.state_dict()}, args.save_dir + '/trn_inta_nn_' + args.exp_id)
    print('ep:', ep)
    print('train_loss:', train_loss)
    print('best_loss_trn, best_epoch_trn:', best_loss_trn, best_ep_trn)

    eval_loss = eval()
    if eval_loss < best_loss_ts:
        best_loss_ts = eval_loss
        best_ep_ts = ep
        print('saving models for evaluating')
        torch.save({'model_state_dict': inta_nn.state_dict()}, args.save_dir + '/ts_inta_nn_' + args.exp_id)
    print('eval_loss:', eval_loss)
    print('best_loss_ts, best_epoch_ts:', best_loss_ts, best_ep_ts)
