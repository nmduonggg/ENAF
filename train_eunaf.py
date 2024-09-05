"""
Train EUNAF-version of backbone SISR network
"""

import os
import torch
import numpy as np
import torch.utils.data as torchdata
import torch.nn.functional as F
import torch.nn as nn
import tqdm
import wandb

#custom modules
import data
import evaluation
import loss
import model as supernet
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import utils
from option import parser
from template import train_template as template


args = parser.parse_args()

if args.template is not None:
    template.set_template(args)

print('[INFO] load trainset "%s" from %s' % (args.trainset_tag, args.trainset_dir))
trainset = data.load_trainset(args)
XYtrain = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)

n_sample = len(trainset)
print('[INFO] trainset contains %d samples' % (n_sample))

# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=4)

cost_dict = {
    'srresnet': [0, 2.04698, 3.66264, 5.194], 
    'fsrcnn': [0, 146.42, 315.45, 468.2]
}
baseline_cost_dict = {
    'srresnet': 5.194,
    'fsrcnn': 468.2
}

cost_ees = cost_dict[args.backbone_name]
baseline_cost = baseline_cost_dict[args.backbone_name]

# model
arch = args.core.split("-")
name = args.template
core = supernet.config(args)
if args.weight:
    loaded_state_dict = torch.load(args.weight, map_location=torch.device('cpu'))
    # new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
    
    print(f"[INFO] Load weight from {args.weight}")
    core.load_state_dict(loaded_state_dict, strict=True)

# utils.calc_flops(core, (1, 3, 32, 32))
    
core.cuda()

# initialization
lr = args.lr
batch_size = args.batch_size
epochs = args.max_epochs - args.start_epoch
num_blocks = args.n_resgroups if args.n_resgroups > 0 else args.n_resblocks
num_blocks = min(num_blocks//2, args.n_estimators)

num_blocks = 4
    
# for 2nd training stage
ee_params, tail_params = list(), list()
for n, p in core.named_parameters():
    if p.requires_grad:
        if 'estimator' in n or 'predictor' in n:
            ee_params.append(p) 
        else:
            tail_params.append(p)

optimizer = Adam([
        {"params": ee_params}, 
        {"params": tail_params, 'lr': 1e-7}
    ], lr=lr, weight_decay=args.weight_decay)

lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
early_stopper = utils.EarlyStopper(patience=15)
loss_func = loss.create_loss_func(args.loss)

# working dir
fname = name+f'_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_ng{args.n_resgroups}_st{args.train_stage}' if args.n_resgroups > 0 \
    else name+f'_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_st{args.train_stage}'
out_dir = os.path.join(args.cv_dir, fname)
os.makedirs(out_dir, exist_ok=True)
print("Load ckpoint to: ", out_dir)

def loss_l1s(yfs, yt):
    l1_loss = 0
    for i, yf in enumerate(yfs[1:]):    # skip bicubic
        l1_loss += loss_func(yf, yt)
        # if i==len(yfs)-2:
        #     l1_loss += loss_func(yf, yt)
    return l1_loss

def loss_unc_psnr_1est(yfs, masks, yt):
    """
    yfs: list of yf [BxCxHxW]x3
    masks: Bx3
    yt: GT
    """
    uscale = 1
    yfs = [yf.clone() for yf in yfs]
    psnrs = torch.stack([evaluation.calculate(args, yf, yt) for yf in yfs[1:]], dim=-1) # Bx3
    psnr_loss = 0.0
    
    for i in range(len(yfs)):
        psnr0 = evaluation.calculate(args, yfs[i], yt).unsqueeze(1)
        mask0 = torch.where(psnr0 != float("inf"), 1., 0.).bool()
        psnr0 = psnr0[mask0]
        # print(psnrs)
        
        mask_0_ = masks[..., i:i+1][mask0]
        psnr_0_loss = loss_func(mask_0_, -psnr0/uscale)
        
        psnr_loss += psnr_0_loss
    psnr_loss /= len(yfs)
    
    return psnr_loss

# training
def train():
    
    # init wandb
    if args.wandb:
        wandb.login(key="")
        wandb.init(
            project='EUNAF',
            group=f'x{args.scale}_UncPsnr_FINAL',
            name=fname, 
            config=vars(args)
        )
    
    step_counter = 0
    best_perf = -1e9 # psnr
    
    cost_ee = np.array(cost_dict[args.backbone_name]) / baseline_cost_dict[args.backbone_name]
    cost_ee = torch.tensor(cost_ee).unsqueeze(0)    # 1xN
    
    if args.train_stage == 1:
        core.freeze_backbone()  # freeze except estimator
    elif args.train_stage == 2:
        core.unfreeze_predictors()
    
    best_val_loss = 1e9
    for epoch in range(epochs):
        track_dict = {}
        
        # start training 
        total_loss = 0.0
        perfs = [0 for _ in range(num_blocks)]
        uncertainty = [0 for _ in range(num_blocks)]
        
        core.train()
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtrain), total=len(XYtrain)):
            
            ### Start testing ###
            if (step_counter % args.val_each_step == 0):
                print(f"Testing at epoch {epoch} and iterations {step_counter}")
                perfs_val = [0 for _ in range(num_blocks)]
                total_val_loss = 0.0
                perf_fused = 0.0
                uncertainty = [0 for _ in range(num_blocks)]
                #walk through the test set
                core.eval()
                for m in core.modules():
                    if hasattr(m, '_prepare'):
                        m._prepare()
                for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
                    x  = x.cuda()
                    yt = yt.cuda()

                    with torch.no_grad():
                        out = core.eunaf_forward(x)
                    
                    outs_mean, masks = out
                    perf_layers_mean = [evaluation.calculate(args, yf, yt) for yf in outs_mean]
                    mask0 = torch.where(perf_layers_mean[0] != float("inf"), 1., 0.).bool()
                    perf_layers_mean[0] = perf_layers_mean[0][mask0]
                    
                    if args.train_stage==0:
                        val_loss = loss_l1s(outs_mean, yt)
                        # val_loss = loss_func(outs_mean[-1], yt)
                    elif args.train_stage==1:
                        val_loss = loss_unc_psnr_1est(outs_mean, masks, yt)
                        
                    total_val_loss += val_loss.item() if torch.is_tensor(val_loss) else val_loss
                    
                    for i, p in enumerate(perf_layers_mean):
                        perfs_val[i] += p.mean().item()
                        uncertainty[i] += masks[i].cpu().detach().mean().item()

                perfs_val = [p / len(XYtest) for p in perfs_val]
                perf_fused /= len(XYtest)
                print(perfs_val, perf_fused)
                uncertainty = [u / len(XYtest) for u in uncertainty]
                total_val_loss /= len(XYtest)

                for i, p in enumerate(perfs_val):
                    track_dict["val_perf_"+str(i)] = p
                    track_dict["val_unc_"+str(i)] = uncertainty[i]
                    
                track_dict["val_l1_loss"] = total_val_loss

                log_str = f'[INFO] Epoch {epoch} - Val L: {total_val_loss}'
                print(log_str)
                # torch.save(core.state_dict(), os.path.join(out_dir, f'E_%d_P_%.3f.t7' % (epoch, mean_perf_f)))
                
                if best_val_loss > total_val_loss:
                    best_val_loss = total_val_loss
                    torch.save(core.state_dict(), os.path.join(out_dir, '_best.t7'))
                    print('[INFO] Save best performance model %d with performance %.3f' % (epoch, best_val_loss))   
                        
                if args.wandb: wandb.log(track_dict)
                torch.save(core.state_dict(), os.path.join(out_dir, '_last.t7'))
                
                if early_stopper.early_stop(total_val_loss):
                    break
                
            ### End testing ###
            
            # intialize
            x  = x.cuda()
            yt = yt.cuda()
            train_loss = 0.0
            
            # inference
            out = core.eunaf_forward(x)   # outs, density, mask
            outs_mean, masks = out
            
            # freeze batch norm
            for m in core.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            
            perf_layers_mean = [evaluation.calculate(args, yf, yt) for yf in outs_mean]
            
            train_loss = 0.0
            if args.train_stage==0:
                train_loss += loss_l1s(outs_mean, yt)
            elif args.train_stage==1:
                train_loss = loss_unc_psnr_1est(outs_mean, masks, yt)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_loss += train_loss.item() if torch.is_tensor(train_loss) else train_loss
            
            for i, p in enumerate(perf_layers_mean):
                perfs[i] = perfs[i] + p.mean().item()
                uncertainty[i] = uncertainty[i] + (masks[i]).detach().cpu().mean()
                
            step_counter += 1
                
                
        lr_scheduler.step()
        total_loss /= len(XYtrain)
        perfs = [p / len(XYtrain) for p in perfs]
        uncertainty = [u / len(XYtrain) for u in uncertainty]
        
        for i, p in enumerate(perfs):
            track_dict["perf_"+str(i)] = p
        track_dict["train_loss"] = total_loss
        
        log_str = '[INFO] E: %d | LOSS: %.3f | Uncertainty: %.3f' % (epoch, total_loss, uncertainty[-1])
        print(log_str)
        
        
if __name__=='__main__':
    train()