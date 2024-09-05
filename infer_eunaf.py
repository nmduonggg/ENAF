import os
import torch
import torch.utils.data as torchdata
import torch.nn.functional as F
import tqdm
import wandb
import cv2
import matplotlib.pyplot as plt
import numpy as np

#custom modules
import data
import evaluation
import loss
import model as supernet
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import utils
from option import parser
from template import test_template as template


args = parser.parse_args()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if args.template is not None:
    template.set_template(args)

# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=0)

cost_dict = {
    'srresnet': [0, 2.04698, 3.66264, 5.194], 
    'fsrcnn': [0, 146.42, 315.45, 468.2],
    'carn': [0, 778.55, 868.86, 1161.72]
}
baseline_cost_dict = {
    'srresnet': 5.194,
    'fsrcnn': 468.2,
    'carn': 1162.72
}


cost_ees = cost_dict[args.backbone_name]
baseline_cost = baseline_cost_dict[args.backbone_name]

# model
arch = args.core.split("-")
name = args.template
core = supernet.config(args)
if args.weight:
    fname = name + f"_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_ng{args.n_resgroups}_st{args.train_stage}" if args.n_resgroups > 0 \
        else name + f"_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_st{args.train_stage}"
    out_dir = os.path.join(args.cv_dir, 'jointly_nofreeze', 'Error-predict', '1est', fname)
    args.weight = os.path.join(out_dir, '_best.t7')
    print(f"[INFO] Load weight from {args.weight}")
    core.load_state_dict(torch.load(args.weight), strict=True)
core.cuda()

loss_func = loss.create_loss_func(args.loss)

# working dir
# out_dir = os.path.join(args.analyze_dir, args.template, name+f'_nblock{args.nblocks}', args.testset_tag)
out_dir = os.path.join(args.analyze_dir, "by_patches", name+f'_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_st{args.train_stage}', args.testset_tag)
print('Load to: ', out_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
def gray2heatmap(image):
    heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    
    return heatmap

def fuse_classified_patch_level(p_yfs, p_masks, im_idx, eta, visualize=False, imscore=None):
    # p_yfs, p_masks: all patches of yfs and masks of all num block stages [[HxWxC]*n_patches]xnum_blocks
    yfs = [np.stack(pm, axis=0) for pm in p_yfs]  # PxHxWxC
    masks = [np.concatenate(pm, axis=0) for pm in p_masks] # Bx3
    per_class = 0
    if imscore is not None:
        imscores = np.array(imscore) # N
        q1, q2, q3 = np.percentile(imscore, [25, 50, 75])
        
        q1 = 10
        p0 = (imscores <= q1).astype(int)
        p1 = (np.logical_and(q1 < imscores, imscores <= q2)).astype(int)
        p2 = (np.logical_and(q2 < imscores, imscores <= q3)).astype(int)
        p3 = (q3 < imscores).astype(int)
        
        per_class = torch.tensor(np.stack([p0*1.2, p1*0.00, p2*0.00, p3*0.00], axis=-1))   # PxN
    
    costs = np.array(cost_ees)
    costs = (costs - costs.min()) / (costs.max() - costs.min())
    # costs = costs / costs.max()
    # normalized_masks = np.stack(masks, axis=-1) 
    
    normalized_masks = np.stack(masks, axis=1) 
    # normalized_masks = (normalized_masks - np.min(normalized_masks, axis=-1, keepdims=True)) / (np.max(normalized_masks, axis=-1, keepdims=True) - np.min(normalized_masks, axis=-1, keepdims=True))
    normalized_masks = normalized_masks + eta*costs.reshape(1, -1)
    masks = [
        normalized_masks[:, i] for i in range(len(yfs)) # PxN
    ]
    
    all_masks = torch.tensor(np.stack(masks, axis=-1)) # P -> PxN
    
    all_masks -= per_class
    
    raw_indices = torch.argmin(all_masks, dim=-1)    # 0->N-1, P
    onehot_indices = F.one_hot(raw_indices, num_classes=len(masks)).float() # PxN
    
    processed_outs = 0
    
    percents = list()
    for i in range(len(p_masks)):
        fout = yfs[i] 
        cur_mask = onehot_indices[..., i].numpy().astype(np.uint8)
        percents.append(cur_mask.mean())
        cur_mask = cur_mask.reshape(-1, 1, 1, 1)
        
        cur_fout = (fout*cur_mask)
        processed_outs += cur_fout
    
    fused_classified = [processed_outs[i,...] for i in range(processed_outs.shape[0])]
    
    if visualize:
        class_colors = [
            [0, 0, 255],    # blue
            [19, 239, 85],  # green
            [235, 255, 128],    # yellow
            [255, 0, 0] # red
        ]
    
        processed_colors = 0
        
        for i in range(len(p_masks)):
            fout = np.array(class_colors[i]).reshape(1, 1, 1, -1)   # BCHW
            fout = fout * np.ones_like(yfs[i])
            fout[:, :1, :, :] = 0
            fout[:, -1:, :, :] = 0
            fout[:, :, :1, :] = 0
            fout[:, :, -1:, :] = 0
            
            cur_mask = onehot_indices[..., i].numpy().astype(np.uint8)
            cur_mask = cur_mask.reshape(-1, 1, 1, 1)
            cur_fout = fout * cur_mask
            
            processed_colors += cur_fout
        
        fused_colors = [processed_colors[i,...] for i in range(processed_colors.shape[0])]
        
        return fused_classified, percents, fused_colors
            
    return fused_classified, percents
        
# testing

t = 5e-3
psnr_unc_map = np.ones((len(XYtest), 12))
num_blocks = args.n_resgroups // 2 if args.n_resgroups > 0 else args.n_resblocks // 2 
num_blocks = min(args.n_estimators, num_blocks)
num_blocks = 4

patch_size = 32
step = 28
alpha = 0.7

def test(eta):
    psnrs_val = [0 for _ in range(num_blocks)]
    ssims_val = [0 for _ in range(num_blocks)]
    uncertainty_val = [0 for _ in range(num_blocks)]
    total_val_loss = 0.0
    total_mask_loss = 0.0
    psnr_fuse, ssim_fuse = 0.0, 0.0
    psnr_fuse_err, ssim_fuse_err = 0.0, 0.0
    psnr_fuse_unc, ssim_fuse_unc = 0.0, 0.0
    psnr_fuse_auto, ssim_fuse_auto = 0.0, 0.0
    
    # for visualization
    outdir = 'visualization/'
    patch_dir = os.path.join(outdir, 'patches')
    os.makedirs(patch_dir, exist_ok=True)
    
    #walk through the test set
    core.eval()
    for m in core.modules():
        if hasattr(m, '_prepare'):
            m._prepare()
            
    percent_total = np.zeros(shape=[num_blocks])
    percent_total_err = np.zeros(shape=[num_blocks])
    percent_total_auto = np.zeros(shape=[num_blocks])
    
    test_patch_psnrs = list()
    
    real_and_preds = {
        'imscore': [],
        'pred_0': [], 'pred_1': [],'pred_2': [], 'pred_3': [],
        'real_0': [], 'real_1': [],'real_2': [], 'real_3': []
    }
    
    cnt = 0
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
        # if batch_idx != 58: continue
        # if cnt > 20: break
        cnt += 1

        torch.manual_seed(0)
        # x = x + torch.randn_like(x) * 0.01
        
        yt = yt.squeeze(0).permute(1,2,0).cpu().numpy()
        yt = utils.modcrop(yt, args.scale)
        yt = torch.tensor(yt).permute(2,0,1).unsqueeze(0)
        
        # cut patches
        x_np = x.permute(0,2,3,1).squeeze(0).numpy()
        lr_list, num_h, num_w, h, w = utils.crop_cpu(x_np, patch_size, step)
        # yt = yt[:, :, :h*args.scale, :w*args.scale]
        y_np = yt.permute(0,2,3,1).squeeze(0).numpy()
        hr_list = utils.crop_cpu(y_np, patch_size * args.scale, step*args.scale)[0]
        yt = yt[:, :, :h*args.scale, :w*args.scale]
        
        combine_img_lists = []
        fusion_outputs = []
        current_imscore = []
        
        all_imgs, all_gts = [], []
        for pid, (lr_img, hr_img) in enumerate(zip(lr_list, hr_list)):
            img = lr_img.astype(np.float32) 
            img = img[:, :, :3]
            gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)   
            laplac = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
            imscore = cv2.convertScaleAbs(laplac).mean()
            
            real_and_preds['imscore'].append(imscore)
            # current_imscore.append(imscore)
            
            img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
            
            gt = hr_img.astype(np.float32) 
            gt = gt[:, :, :3]
            gt = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0)
            
            img = img.to(device)
            gt = gt.to(device)
            with torch.no_grad():
                out = core.eunaf_infer(img, eta, imscore)
            
                combine_img_lists.append(out[0].cpu().permute(1,2,0).numpy())
                fusion_outputs.append(out[0].cpu())
                # combine_unc_lists[i].append(p_masks[:, i].cpu())
            
        # masks = [p_masks[:, i].detach() for i in range(p_masks.shape[1])]
        yf = torch.from_numpy(utils.combine(combine_img_lists, num_h, num_w, h, w, patch_size, step, args.scale)).permute(2,0,1).unsqueeze(0)
            
        psnr, ssim = evaluation.calculate_all(args, yf, yt)
        
        #### FUSION START HERE ####
        visualize = False
        # fusion_outputs = fuse_classified_patch_level(combine_img_lists, combine_unc_lists, batch_idx, eta, visualize=visualize, imscore=current_imscore)
        
        # if len(fusion_outputs) == 2:
        #     fused_auto_patches, percents_auto = fusion_outputs
        # else:
        #     fused_auto_patches, percents_auto, fused_color_map = fusion_outputs
        
        patch_psnr_1_img = list()
        for patch_f, patch_t in zip(fusion_outputs, hr_list):
            patch_t = torch.tensor(patch_t).permute(2,0,1).unsqueeze(0) 
            psnr_patch = evaluation.calculate(args, patch_f, patch_t)
            patch_psnr_1_img.append(psnr_patch.mean())
            
        test_patch_psnrs += patch_psnr_1_img
        
        psnr_fuse_auto += psnr
        ssim_fuse_auto += ssim
    
    percent = np.array(core.counts) / np.sum(core.counts)
    
    auto_flops = np.sum(percent * np.array(cost_ees))
    summary_percent = (auto_flops / baseline_cost)*100
    print(f"Percent FLOPS: {auto_flops} - {summary_percent}")
    
    psnr_fuse_auto /= len(XYtest)
    ssim_fuse_auto /= len(XYtest)
    
    print("Avg patch PSNR: ", np.mean(np.array(test_patch_psnrs)))
    
    print("fusion auto psnr: ", psnr_fuse_auto)
    print("fusion auto ssim: ", ssim_fuse_auto)
    print("Sampling patches rate:")
    for perc in percent:
        print( f"{(perc*100):.3f}", end=' ')
    
    return real_and_preds, auto_flops, psnr_fuse_auto
    
if __name__ == '__main__':
    # get 1 patch flops
    utils.calc_flops(core, (1, 3, 32, 32))
    
    etas, flops, psnrs = [], [], []
    for eta in np.linspace(0.15, 5.0, 15):
        print("="*20, f"eta = {eta}", "="*20)
        real_and_preds, auto_flops, psnr_fuse_auto = test(eta)
        etas.append(eta)
        flops.append(auto_flops) #
        psnrs.append(psnr_fuse_auto)
        
        # break
    print(etas)
    print(flops)
    print(psnrs)