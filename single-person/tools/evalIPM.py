from torch.utils.data import DataLoader
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
from utils.utils import cal_loss_sgl_ipm
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def eval():
    total_loss, total_num_mean = 0, 0
    total_loss_x, total_loss_y, total_loss_the, total_loss_phi = 0, 0, 0, 0
    total_loss_x_dot, total_loss_y_dot, total_loss_the_dot, total_loss_phi_dot = 0, 0, 0, 0
    ct = 1
    for batch in evalLoader:
        expName = 'exp' + str(ct)
        mainPaths = [args.prntDir + '/blender/c3d_ipm/ipm_indiv_pred/' + expName + '/',
                     args.prntDir + '/blender/points_ipm/ipm_indiv_pred/' + expName + '/']
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
        pred, pred_dot= person.output()
        person.save('0', mainPaths)
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
        ct += 1
    total_loss /= total_num_mean
    total_loss_x /= total_num_mean
    total_loss_y /= total_num_mean
    total_loss_the /= total_num_mean
    total_loss_phi /= total_num_mean
    total_loss_x_dot /= total_num_mean
    total_loss_y_dot /= total_num_mean
    total_loss_the_dot /= total_num_mean
    total_loss_phi_dot /= total_num_mean
    return total_loss, total_loss_x, total_loss_y, total_loss_the, total_loss_phi

args = parse_args()
args.prntDir = prnt_dir
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
print(device)
args.device = device
pd_params = {'x': args.ks_x.to(device), 'y': args.ks_y.to(device), 'theta': args.ks_theta.to(device), 'phi': args.ks_phi.to(device)}

evalSet = IpmSDataset('eval', args)
evalLoader = DataLoader(evalSet, batch_size=args.bat_sz, num_workers=args.num_workers,
                        drop_last=False, shuffle=False)

#Load individual-level IPM models
modelFsnn = LSTM_force_cell(input_size=args.snInputSz, embedding_size=args.snEmdSz, rnn_size=args.snRnnSz, output_size=args.snOutptSz)
modelRod = NNRod(input_dim=args.rodInputSz, output_dim=args.rodOutptSz, hidden_size=args.rodHdSz)
modelFsnn = modelFsnn.double().to(args.device)
modelRod = modelRod.double().to(args.device)

modelFsnnPath = prnt_dir + args.modelFsnnPath
modelRodPath = prnt_dir + args.modelRodPath
ckptFsnn = torch.load(modelFsnnPath, map_location=args.device)
ckptRod = torch.load(modelRodPath, map_location=args.device)
modelFsnn.load_state_dict(ckptFsnn['model_state_dict'])
modelRod.load_state_dict(ckptRod['model_state_dict'])
miu_ground = ckptFsnn['physics']
modelFsnn.eval()
modelRod.eval()

eval_loss, loss_x, loss_y, loss_the, loss_phi = eval()
print(eval_loss, loss_x, loss_y, loss_the, loss_phi)
print()
