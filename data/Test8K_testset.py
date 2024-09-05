import tqdm
from torch.utils.data import Dataset
from .common import load_image_as_Tensor

class Test8K_testset(Dataset):
    def __init__(self, root, scale=4, style='RGB', rgb_range=1.0):
        super(Test8K_testset, self).__init__()
        assert scale==4, "Only support scale=4 for Test2K"
    
        self.N_raw_image = 100
        self.N = self.N_raw_image

        self.X, self.Y = [], []
        self.scale = scale
        self.root = root
        self.style = style
        self.rgb_range = rgb_range
        
        # for i in tqdm.tqdm(range(self.N_raw_image)):
        #     fn = str(1301+i) + '.png'
        #     X_im_file_name = self.root + 'LR/' +f'X{self.scale}/' + fn
        #     X_data = load_image_as_Tensor(X_im_file_name, style, rgb_range)
        #     self.X.append(X_data)

        #     Y_im_file_name = self.root + 'HR/X4/' + fn
        #     Y_data = load_image_as_Tensor(Y_im_file_name, style, rgb_range)
        #     self.Y.append(Y_data)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        fn = str(1401+idx) + '.png'
        X_im_file_name = self.root + 'LR/' + f'X{self.scale}/' + fn  
        X_data = load_image_as_Tensor(X_im_file_name, self.style, self.rgb_range)
        
        Y_im_file_name = self.root + 'HR/X4/' + fn
        Y_data = load_image_as_Tensor(Y_im_file_name, self.style, self.rgb_range)

        return X_data, Y_data