import numpy as np
import torch
import skimage.color as sc
import imageio
import random

def get_patch(im_lr, im_hr, lr_patch_size=21, scale=2):
    lw = im_lr.shape[2]
    lh = im_lr.shape[1]

    ix = random.randrange(0, lw - lr_patch_size + 1)
    iy = random.randrange(0, lh - lr_patch_size + 1)

    im_lr_patch = im_lr[:, iy      : iy+lr_patch_size,        ix      : ix+lr_patch_size       ]
    im_hr_patch = im_hr[:, iy*scale:(iy+lr_patch_size)*scale, ix*scale:(ix+lr_patch_size)*scale]

    return im_lr_patch, im_hr_patch

def load_image_as_Tensor(im_file_name, style='RGB', rgb_range=1.0):
    data = imageio.imread(im_file_name)

    if style == 'RGB':
        if data.ndim == 2:
            data = np.stack((data,)*3, axis=-1)
    elif style == 'Y':
        if data.ndim == 2:
            data = np.expand_dims(data, axis=2)
        
        if data.shape[2] == 3:
            data = np.expand_dims(sc.rgb2ycbcr(data)[:, :, 0], 2)
    else:
        print('[ERRO] unknown style; should be Y or RGB')
        assert(0)

    data = np.ascontiguousarray(np.transpose(data, (2,0,1)))
    data = torch.Tensor(data).float().mul_(rgb_range/255.)

    return data


def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(a) for a in args]