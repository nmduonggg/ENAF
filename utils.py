import os
import torch
import cv2
import shutil
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import OrderedDict
import torch.nn.functional as F

import warnings
from calflops import calculate_flops
from ptflops import get_model_complexity_info

def calc_flops(model, size):
    flops, macs, params = calculate_flops(model=model, 
                                        input_shape=size,
                                        output_as_string=True,
                                        print_detailed=True,
                                        output_precision=4)
    print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    
def calc_flops_2blocks(model, size):
    if len(size)>3:
        size = size[1:]
    macs, params = get_model_complexity_info(model, size, as_strings=True, backend='aten',
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

def confidence_interval_mean(estimated_mean, estimated_stddev, sample_size, confidence_level):
    """
    Calculate the confidence interval for the population mean.
    
    Parameters:
        estimated_mean (float): The estimated mean of the sample.
        estimated_stddev (float): The estimated standard deviation of the sample.
        sample_size (int): The size of the sample.
        confidence_level (float): The desired confidence level (e.g., 0.95 for 95% confidence).
    
    Returns:
        tuple: A tuple containing the lower and upper bounds of the confidence interval.
    """
    # Calculate the critical value based on the confidence level
    z_critical = stats.norm.ppf((1 + confidence_level) / 2)  # Two-tailed test
    
    # Calculate the margin of error
    margin_of_error = z_critical * (abs(estimated_stddev) / (sample_size ** 0.5))
    
    # Calculate the confidence interval
    lower_bound = estimated_mean - margin_of_error
    upper_bound = estimated_mean + margin_of_error
    
    return lower_bound, upper_bound

def resize_image_tensor(lr_image, hr_image, scale, rgb_range):
    hr_height, hr_width = hr_image.shape[2:4]
    lr_height, lr_width = lr_image.shape[2:4]
    if lr_height * scale != hr_height or lr_width * scale != lr_width:
        hr_width = lr_width * scale
        hr_height = lr_height * scale
        hr_image = F.interpolate(hr_image, size=(hr_height, hr_width), mode='bicubic')
        hr_image.clamp(min=0, max = rgb_range)
    return hr_image

def crop_cpu(img, crop_sz, step):
    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list=[]
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            lr_list.append(crop_img)
    h=x + crop_sz
    w=y + crop_sz
    return lr_list, num_h, num_w, h, w

def combine(sr_list, num_h, num_w, h, w, patch_size, step, scale):
    index=0
    sr_img = np.zeros((h*scale, w*scale, 3), 'float32')
    for i in range(num_h):
        for j in range(num_w):
            sr_img[i*step*scale:i*step*scale+patch_size*scale, j*step*scale:j*step*scale+patch_size*scale,:] += sr_list[index]
            index+=1
    sr_img=sr_img.astype('float32')

    for j in range(1,num_w):
        sr_img[:,j*step*scale:j*step*scale+(patch_size-step)*scale,:]/=2

    for i in range(1,num_h):
        sr_img[i*step*scale:i*step*scale+(patch_size-step)*scale,:,:]/=2
    return sr_img

def combine_addmask(sr_list, color_list, num_h, num_w, h, w, patch_size, step, scale):
    index=0
    sr_img = np.zeros((h*scale, w*scale, 3), 'float32')
    color_raws = np.zeros((num_h, num_w, 3))
    for i in range(num_h):
        for j in range(num_w):
            sr_img[i*step*scale:i*step*scale+patch_size*scale, j*step*scale:j*step*scale+patch_size*scale,:] += sr_list[index]
            color_raws[i, j, :] = color_list[index][10, 10, :]    # take color only
            index+=1
    sr_img=sr_img.astype('float32')

    for j in range(1,num_w):
        sr_img[:,j*step*scale:j*step*scale+(patch_size-step)*scale,:]/=2

    for i in range(1,num_h):
        sr_img[i*step*scale:i*step*scale+(patch_size-step)*scale,:,:]/=2
        
    sr_img = (sr_img * 255).astype(np.uint8)
    sr_img = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
        
    index2 = 0
    for i in range(num_h):
        for j in range(num_w):
            # add mask
            alpha = 1
            beta = 0.35
            gamma = 0
            bbox1 = [j * step * scale + 8, i * step * scale + 8,
                    j * step * scale + patch_size * scale - 9,
                    i * step * scale + patch_size * scale - 9]  # xl,yl,xr,yr
            zeros1 = np.zeros((sr_img.shape), 'float32')
            mask2 = cv2.rectangle(zeros1, (bbox1[0]+1, bbox1[1]+1), (bbox1[2]-1, bbox1[3]-1), color=list(color_raws[i, j, :]), thickness=-1)# hard red
            mask2 = mask2.astype(np.uint8)
            sr_img = cv2.addWeighted(sr_img, alpha, mask2, beta, gamma)
            index2+=1
            
    sr_img = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)
    return sr_img

def apply_alpha_mask(foreground, background, alpha):
    # Convert uint8 to float
    foreground = (foreground.astype(float)*255).astype(np.uint8)
    background = (background.astype(float)*255).astype(np.uint8)

    # Multiply the foreground with the alpha matte
    # foreground = cv2.multiply(alpha, foreground)
    
    # Multiply the background with (1 - alpha)
    # background = cv2.multiply(1.0 - alpha, background)
    
    # Add the masked foreground and background.
    # outImage = cv2.add(foreground, background)
    outImage = cv2.addWeighted(foreground, alpha, background, beta=0.2, gamma=0)
    # Return a normalized output image for display
    return outImage

layer_modules = (
    nn.Conv2d, nn.ConvTranspose2d,
    nn.Linear,
    nn.BatchNorm2d,
)

def summary(model, input_size, *args, **kwargs):
    """Summarize the given input model.
    Summarized information are 1) output shape, 2) kernel shape,
    3) number of the parameters and 4) operations (Mult-Adds)
    Args:
        model (Module): Model to summarize
        x (Tensor): Input tensor of the model with [N, C, H, W] shape
                    dtype and device have to match to the model
        args, kwargs: Other argument used in `model.forward` function
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    x = torch.zeros(input_size).to(next(model.parameters()).device)

    def register_hook(module):
        def hook(module, inputs, outputs):
            cls_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            key = None
            for name, item in module_names.items():
                if item == module:
                    key = "{}_{}".format(module_idx, name)
                    break
            assert key

            info = OrderedDict()
            info["id"] = id(module)
            if isinstance(outputs, (list, tuple)):
                try:
                    info["out"] = list(outputs[0].size())
                except AttributeError:
                    info["out"] = list(outputs[0].data.size())
            else:
                info["out"] = list(outputs.size())

            info["ksize"] = "-"
            info["inner"] = OrderedDict()
            info["params_nt"], info["params"], info["macs"] = 0, 0, 0
            for name, param in module.named_parameters():
                info["params"] += param.nelement() * param.requires_grad
                info["params_nt"] += param.nelement() * (not param.requires_grad)

                if name == "weight":
                    ksize = list(param.size())
                    if len(ksize) > 1:
                        ksize[0], ksize[1] = ksize[1], ksize[0]
                    info["ksize"] = ksize

                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                        assert len(inputs[0].size()) == 4 and len(inputs[0].size()) == len(outputs[0].size())+1

                        in_c, in_h, in_w = inputs[0].size()[1:]
                        k_h, k_w = module.kernel_size
                        out_c, out_h, out_w = outputs[0].size()
                        groups = module.groups
                        kernel_mul = k_h * k_w * (in_c // groups)

                        kernel_mul_group = kernel_mul * out_h * out_w * (out_c // groups)
                        total_mul = kernel_mul_group * groups
                        info["macs"] += 2 * total_mul


                    elif isinstance(module, nn.BatchNorm2d):
                        info["macs"] += inputs[0].size()[1]
                    else:
                        info["macs"] += param.nelement()

                elif "weight" in name:
                    info["inner"][name] = list(param.size())
                    info["macs"] += param.nelement()

            if list(module.named_parameters()):
                for v in summary.values():
                    if info["id"] == v["id"]:
                        info["params"] = "(recursive)"

            if info["params"] == 0:
                info["params"], info["macs"] = "-", "-"

            summary[key] = info

        if isinstance(module, layer_modules) or not module._modules:
            hooks.append(module.register_forward_hook(hook))



    module_names = get_names_dict(model)

    hooks = []
    summary = OrderedDict()

    model.apply(register_hook)
    try:
        with torch.no_grad():
            model(x) if not (kwargs or args) else model(x, *args, **kwargs)
    finally:
        for hook in hooks:
            hook.remove()
    # Use pandas to align the columns
    df = pd.DataFrame(summary).T


    df["Mult-Adds"] = pd.to_numeric(df["macs"], errors="coerce")
    df["Params"] = pd.to_numeric(df["params"], errors="coerce")
    df["Non-trainable params"] = pd.to_numeric(df["params_nt"], errors="coerce")
    df = df.rename(columns=dict(
        ksize="Kernel Shape",
        out="Output Shape",
    ))
    # with warnings.catch_warnings():
    #     warnings.filterwarnings('ignore')
    #     df_sum = df.sum()


    df.index.name = "Layer"

    df = df[["Kernel Shape", "Output Shape", "Params", "Mult-Adds"]]
    max_repr_width = max([len(row) for row in df.to_string().split("\n")])

    return df["Mult-Adds"], df

def get_names_dict(model):
    """Recursive walk to get names including path."""
    names = {}

    def _get_names(module, parent_name=""):
        for key, m in module.named_children():
            cls_name = str(m.__class__).split(".")[-1].split("'")[0]
            num_named_children = len(list(m.named_children()))
            if num_named_children > 0:
                name = parent_name + "." + key if parent_name else key
            else:
                name = parent_name + "." + cls_name + "_"+ key if parent_name else key
            names[name] = m

            if isinstance(m, torch.nn.Module):
                _get_names(m, parent_name=name)

    _get_names(model)
    return names

def laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplac = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    mask_img = cv2.convertScaleAbs(laplac)
    return mask_img

def save_hmap(hmap, hmap_x, hmap_y, save_path, bins):
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.matshow(hmap, origin='lower')
    ax.xaxis.set_ticks(np.arange(0,bins,1))
    ax.set_xticklabels(np.around(np.linspace(min(hmap_x), max(hmap_x), num=bins), 2), rotation=90)
    ax.xaxis.set_ticks_position('bottom')

    ax.yaxis.set_ticks(np.arange(0,bins,1))
    ax.set_yticklabels(np.around(np.linspace(min(hmap_y), max(hmap_y), num=bins), 2))

    fig.savefig(save_path, bbox_inches='tight', pad_inches=0.05)

    plt.cla()
    plt.close("all")

def modcrop(img_in, scale):
    """img_in: Numpy, HWC or HW"""
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img



class LrScheduler:
    def __init__(self, optimizer, base_lr, lr_decay_ratio, epoch_step):
        self.base_lr = base_lr
        self.lr_decay_ratio = lr_decay_ratio
        self.epoch_step = epoch_step
        self.optimizer = optimizer

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.base_lr * (self.lr_decay_ratio ** (epoch // self.epoch_step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            if epoch%self.epoch_step==0:
                print('[INFO] Setting learning_rate to %.2E'%lr)
                
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False