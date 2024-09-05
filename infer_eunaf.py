import os
import torch
import torch.utils.data as torchdata
import torch.nn.functional as F
import tqdm
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
    print(f"[INFO] Load weight from {args.weight}")
    core.load_state_dict(torch.load(args.weight), strict=True)
core.cuda()
loss_func = loss.create_loss_func(args.loss)
    
def gray2heatmap(image):
    heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    
    return heatmap
        
# testing
t = 5e-3
num_blocks = 4

patch_size = 32
step = 28
alpha = 0.7

def test(eta):
    total_val_loss = 0.0
    total_mask_loss = 0.0
    psnr_fuse_auto = 0.0
    
    # for visualization
    outdir = 'visualization/'
    patch_dir = os.path.join(outdir, 'patches')
    os.makedirs(patch_dir, exist_ok=True)
    
    #walk through the test set
    core.eval()
    for m in core.modules():
        if hasattr(m, '_prepare'):
            m._prepare()
    
    test_patch_psnrs = list()
    
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
                
        yf = torch.from_numpy(utils.combine(combine_img_lists, num_h, num_w, h, w, patch_size, step, args.scale)).permute(2,0,1).unsqueeze(0)
            
        psnr, _ = evaluation.calculate_all(args, yf, yt)
        
        patch_psnr_1_img = list()
        for patch_f, patch_t in zip(fusion_outputs, hr_list):
            patch_t = torch.tensor(patch_t).permute(2,0,1).unsqueeze(0) 
            psnr_patch = evaluation.calculate(args, patch_f, patch_t)
            patch_psnr_1_img.append(psnr_patch.mean())
            
        test_patch_psnrs += patch_psnr_1_img
        
        psnr_fuse_auto += psnr
    
    percent = np.array(core.counts) / np.sum(core.counts)
    
    auto_flops = np.sum(percent * np.array(cost_ees))
    summary_percent = (auto_flops / baseline_cost)*100
    print(f"Percent FLOPS: {auto_flops} - {summary_percent}")
    
    psnr_fuse_auto /= len(XYtest)
    
    print("Avg patch PSNR: ", np.mean(np.array(test_patch_psnrs)))
    
    print("fusion auto psnr: ", psnr_fuse_auto)
    print("Sampling patches rate:")
    for perc in percent:
        print( f"{(perc*100):.3f}", end=' ')
    
    return auto_flops, psnr_fuse_auto
    
if __name__ == '__main__':
    # get 1 patch flops
    utils.calc_flops(core, (1, 3, 32, 32))
    
    etas, flops, psnrs = [], [], []
    for eta in np.linspace(0.15, 5.0, 15):
        
        print("="*20, f"eta = {eta}", "="*20)
        auto_flops, psnr_fuse_auto = test(eta)
        etas.append(eta)
        flops.append(auto_flops) #
        psnrs.append(psnr_fuse_auto)
        
        # break
    print(etas)
    print(flops)
    print(psnrs)