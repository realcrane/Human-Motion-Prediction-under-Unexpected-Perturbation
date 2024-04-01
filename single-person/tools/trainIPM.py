from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import sys
import os
cur_dir = os.getcwd().replace('\\', '/')
prnt_dir = os.path.abspath(os.path.join(cur_dir, os.pardir)).replace('\\', '/')
sys.path.insert(0, prnt_dir)
from cfgs.ipm_single import parse_args
from models.models import *
from data.datasets.ipm_single_dataset import IpmSDataset
from models.funcs import *
from utils.utils import cal_loss_sgl_ipm, log_mul_ipm
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def train():
    modelFsnn.train()
    modelRod.train()
    total_loss, total_num_mean = 0, 0
    total_loss_x, total_loss_y, total_loss_the, total_loss_phi = 0, 0, 0, 0
    total_loss_x_dot, total_loss_y_dot, total_loss_the_dot, total_loss_phi_dot = 0, 0, 0, 0
    for batch in trainLoader:
        train_data = batch['ipmSts'][0].to(args.device)
        mass = batch['mass']
        rodLen = batch['rodLen'][0]
        cartH = batch['cartH'][0]
        outFs = batch['outFs'][0]
        num_frames = train_data.shape[0]
        ini_h_st = torch.zeros(modelFsnn.cell.hidden_size).double().to(args.device).detach()
        ini_c_st = torch.zeros(modelFsnn.cell.hidden_size).double().to(args.device).detach()
        person = IPM3D_sgl(modelFsnn, modelRod, pd_params, mass, rodLen, cartH, miu_ground,
                            args.deltaT, num_frames, train_data[0, :4], args.maxRod, args.minRod, ini_h_st, ini_c_st,
                            train_data[0, 4:], outFs, args.device)
        for _ in range(1, num_frames):
            person.step()
        pred, pred_dot = person.output()
        loss_x, loss_y, loss_the, loss_phi, loss_x_dot, loss_y_dot, loss_the_dot, loss_phi_dot = cal_loss_sgl_ipm(
            train_data, pred, pred_dot)
        num_mean = num_frames
        optimizer.zero_grad()
        loss = loss_x + loss_y + loss_the + loss_phi + loss_x_dot + loss_y_dot + loss_the_dot + args.lambdaPhiDot * loss_phi_dot
        loss.requires_grad_(True)
        loss.backward()
        nn.utils.clip_grad_norm_(modelFsnn.parameters(), 1.0)
        nn.utils.clip_grad_norm_(modelRod.parameters(), 1.0)
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
                total_loss_y_dot, total_loss_the_dot, total_loss_phi_dot, total_num_mean, ep, 'train')
    total_loss /= total_num_mean
    return total_loss


def eval():
    modelFsnn.eval()
    modelRod.eval()
    total_loss, total_num_mean = 0, 0
    total_loss_x, total_loss_y, total_loss_the, total_loss_phi = 0, 0, 0, 0
    total_loss_x_dot, total_loss_y_dot, total_loss_the_dot, total_loss_phi_dot = 0, 0, 0, 0
    for batch in evalLoader:
        eval_data = batch['ipmSts'][0].to(args.device)
        mass = batch['mass']
        rodLen = batch['rodLen'][0]
        cartH = batch['cartH'][0]
        outFs = batch['outFs'][0]
        num_frames = eval_data.shape[0]
        ini_h_st = torch.zeros(modelFsnn.cell.hidden_size).double().to(args.device)
        ini_c_st = torch.zeros(modelFsnn.cell.hidden_size).double().to(args.device)
        person = IPM3D_sgl(modelFsnn, modelRod, pd_params, mass, rodLen, cartH, miu_ground, args.deltaT,
                            num_frames, eval_data[0, :4], args.maxRod, args.minRod, ini_h_st, ini_c_st,
                            eval_data[0, 4:], outFs, args.device)
        for _ in range(1, num_frames):
            person.step()
        pred, pred_dot = person.output()
        loss_x, loss_y, loss_the, loss_phi, loss_x_dot, loss_y_dot, loss_the_dot, loss_phi_dot = cal_loss_sgl_ipm(
            eval_data, pred, pred_dot)
        num_mean = num_frames
        loss = loss_x + loss_y + loss_the + loss_phi + loss_x_dot + loss_y_dot + loss_the_dot + args.lambdaPhiDot * loss_phi_dot
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
                total_loss_y_dot, total_loss_the_dot, total_loss_phi_dot, total_num_mean, ep, 'test')
    total_loss /= total_num_mean
    return total_loss

args = parse_args()
args.prntDir = prnt_dir
args.logDir = args.prntDir + '/' + args.logDir
logger = SummaryWriter(args.logDir)
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
print(device)
args.device = device
pd_params = {'x': args.ks_x.to(device), 'y': args.ks_y.to(device), 'theta': args.ks_theta.to(device), 'phi': args.ks_phi.to(device)}

trainSet = IpmSDataset('train', args)
trainLoader = DataLoader(trainSet, batch_size=args.bat_sz, num_workers=args.num_workers,
                        drop_last=False, shuffle=True)
evalSet = IpmSDataset('eval', args)
evalLoader = DataLoader(evalSet, batch_size=args.bat_sz, num_workers=args.num_workers,
                        drop_last=False, shuffle=False)

#Load individual-level IPM models
modelFsnn = LSTM_force_cell(input_size=args.snInputSz, embedding_size=args.snEmdSz, rnn_size=args.snRnnSz, output_size=args.snOutptSz)
modelRod = NNRod(input_dim=args.rodInputSz, output_dim=args.rodOutptSz, hidden_size=args.rodHdSz)
modelFsnn = modelFsnn.double().to(args.device)
modelRod = modelRod.double().to(args.device)
miu_ground = torch.tensor(100.0).to(args.device)
miu_ground.requires_grad = True
optimizer = optim.Adam([{'params': modelFsnn.parameters()}, {'params': modelRod.parameters()},
                        {'params': [miu_ground]}], lr=args.lr)
best_loss_trn, best_ep_trn = 10e5, 0
best_loss_ts, best_ep_ts = 10e5, 0
trnSavePathFsnn = args.prntDir + args.save_dir + '/trnFsnn_' + args.expId
trnSavePathRod = args.prntDir + args.save_dir + '/trnRod_' + args.expId
tsSavePathFsnn = args.prntDir + args.save_dir + '/tsFsnn_' + args.expId
tsSavePathRod = args.prntDir + args.save_dir + '/tsRod_' + args.expId

for ep in range(args.num_eps):
    train_loss = train()
    if train_loss < best_loss_trn:
        best_loss_trn = train_loss
        best_ep_trn = ep
        print('saving models for training')
        torch.save({'model_state_dict': modelFsnn.state_dict(), 'physics': miu_ground.cpu()}, trnSavePathFsnn)
        torch.save({'model_state_dict': modelRod.state_dict()}, trnSavePathRod)
    print('ep:', ep)
    print('train_loss:', train_loss)
    print('best_loss_trn, best_epoch_trn:', best_loss_trn, best_ep_trn)

    eval_loss = eval()
    if eval_loss < best_loss_ts:
        best_loss_ts = eval_loss
        best_ep_ts = ep
        print('saving models for evaluating')
        torch.save({'model_state_dict': modelFsnn.state_dict(), 'physics': miu_ground.cpu()}, tsSavePathFsnn)
        torch.save({'model_state_dict': modelRod.state_dict()}, tsSavePathRod)
    print('eval_loss:', eval_loss)
    print('best_loss_ts, best_epoch_ts:', best_loss_ts, best_ep_ts)
