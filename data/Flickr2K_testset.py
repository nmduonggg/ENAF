import tqdm
from torch.utils.data import Dataset
from .common import load_image_as_Tensor

class Flickr2K_testset(Dataset):
    def __init__(self, root, scale=4, style='RGB', rgb_range=1.0):
        super(Flickr2K_testset, self).__init__()

        self.N_raw_image = 100
        self.N = self.N_raw_image

        self.X, self.Y = [], []
        self.scale = scale
        self.root = root
        
        for i in tqdm.tqdm(range(self.N_raw_image)):
            fn = str(i+1).zfill(6)+'.png'
            X_im_file_name = self.root + 'Flickr2K_LR_bicubic/' +f'X{self.scale}/' + fn.replace('.png', f'x{self.scale}.png')
            X_data = load_image_as_Tensor(X_im_file_name, style, rgb_range)
            self.X.append(X_data)

            Y_im_file_name = self.root + 'Flickr2K_HR/' + fn
            Y_data = load_image_as_Tensor(Y_im_file_name, style, rgb_range)
            self.Y.append(Y_data)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        im_lr = self.X[idx]
        im_hr = self.Y[idx]

        return im_lr, im_hr