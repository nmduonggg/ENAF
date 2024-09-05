import tqdm
from torch.utils.data import Dataset
from data.common import load_image_as_Tensor, get_patch

class DIV2K_validset(Dataset):
    def __init__(self, root, scale=2, style='RGB', rgb_range=1.0):
        super(DIV2K_validset, self).__init__()

        self.N_raw_image = 100 #801-810
        self.N = self.N_raw_image

        self.X, self.Y = [], []
        self.scale = 4
        self.root = root
        self.style=style
        self.rgb_range = rgb_range

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        X_im_file_name = self.root + 'DIV2K_valid_LR_bicubic/X' + str(self.scale) + '/' + ('%04dx%d.png' % (idx+801,self.scale))
        X_data = load_image_as_Tensor(X_im_file_name, self.style, self.rgb_range)

        Y_im_file_name = self.root + 'DIV2K_valid_HR/' + ('%04d.png' % (idx+801))
        Y_data = load_image_as_Tensor(Y_im_file_name, self.style, self.rgb_range)
        
        return X_data, Y_data