import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import tqdm
from data.common import load_image_as_Tensor, get_patch

class DIV2K_trainset(Dataset):
    def __init__(self, root, max_load=800, lr_patch_size=21, scale=2, style='RGB', preload=0, rgb_range=1.0):
        super(DIV2K_trainset, self).__init__()
        self.root = root
        self.repeat = 20
        if max_load > 0:
            self.N_raw_image = min(max_load, 800) #1-800
        else:
            self.N_raw_image = 800 #1-800
        self.N = self.N_raw_image * self.repeat

        self.X, self.Y = [], []
        self.lr_patch_size = lr_patch_size
        self.scale = scale
        self.style = style
        
        self.preload_id = np.zeros(self.N_raw_image)

        for i in tqdm.tqdm(range(self.N_raw_image)):
            if i >= preload:
                break

            X_im_file_name = root + 'DIV2K_train_LR_bicubic/X' + str(scale) + '/' + ('%04dx%d.png' % (i+1,scale))
            X_data = load_image_as_Tensor(X_im_file_name, style, rgb_range)
            self.X.append(X_data)

            Y_im_file_name = root + 'DIV2K_train_HR/' + ('%04d.png' % (i+1))
            Y_data = load_image_as_Tensor(Y_im_file_name, style, rgb_range)
            self.Y.append(Y_data)

            self.preload_id[i] = 1

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        im_idx = idx % self.N_raw_image
        if self.preload_id[im_idx]:
            im_lr = self.X[im_idx]
            im_hr = self.Y[im_idx]
        else:
            X_im_file_name = self.root + 'DIV2K_train_LR_bicubic/X' + str(self.scale) + '/' + ('%04dx%d.png' % (im_idx+1,self.scale))
            im_lr = load_image_as_Tensor(X_im_file_name, self.style)

            Y_im_file_name = self.root + 'DIV2K_train_HR/' + ('%04d.png' % (im_idx+1))
            im_hr = load_image_as_Tensor(Y_im_file_name, self.style)

        im_lr_patch, im_hr_patch = get_patch(im_lr, im_hr, self.lr_patch_size, self.scale)

        if np.random.uniform(0,1) < 0.5:
            im_lr_patch = transforms.functional.hflip(im_lr_patch)
            im_hr_patch = transforms.functional.hflip(im_hr_patch)
            
        if np.random.uniform(0,1) < 0.5:
            im_lr_patch = transforms.functional.vflip(im_lr_patch)
            im_hr_patch = transforms.functional.vflip(im_hr_patch)
            
        if np.random.uniform(0,1) < 0.5:
            im_lr_patch = transforms.functional.rotate(im_lr_patch, 90)
            im_hr_patch = transforms.functional.rotate(im_hr_patch, 90)
            
        return im_lr_patch, im_hr_patch