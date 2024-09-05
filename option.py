import argparse
parser = argparse.ArgumentParser(description="Image Super-Resolution Trainer (clean)", fromfile_prefix_chars="@")

#training hyper-param
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--lr_decay_ratio", type=float, default=0.1, help="lr *= lr_decay_ratio after epoch_steps")
parser.add_argument("--weight_decay",type=float, default=1e-4, help="Weight decay, Default: 1e-4")

parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--batch_size_test", type=int, default=1, help="batch size test")
parser.add_argument("--epoch_step", type=int, default=20, help="epochs after which lr is decayed")
parser.add_argument("--start_epoch", type=int, default=0, help="starting point")
parser.add_argument("--max_epochs", type=int, default=300, help="total epochs to run")
parser.add_argument("--loss", default="L1", help="loss function")
parser.add_argument("--val-each", type=int, default='5', help='Validation each n epochs')
parser.add_argument("--weight", help='Weight path')


# SuperNet hyper-parameters
parser.add_argument("--nblocks", type=int, default=1, help="Number of blocks to be used")
parser.add_argument("--N", type=int, default=-1, help='Number of test instances, only used in testing')
parser.add_argument("--n_estimators", type=int, default=4, help="Number of intermediate uncertainty estimators")

parser.add_argument("--train_stage", type=int, default=0, choices=(0, 1, 2, 3, 4), help="Choose training stage for pretrain backbone, eunaf, align stages")

# Model specifications - EDSR
parser.add_argument('--act', type=str, default='relu',help='activation function')
parser.add_argument('--pre_train', type=str, default='',help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,help='residual scaling')
parser.add_argument('--shift_mean', default=True,help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',choices=('single', 'half'),help='FP precision for test (single | half)')
parser.add_argument('--input_channel', type=int, default=3, help='number of color channels')

# RCAN
parser.add_argument('--n_resgroups', type=int, default=-1, help='number of residual groups in RCAN structures')

# test
parser.add_argument("--visualize", action='store_true')
parser.add_argument("--rgb_channel", action='store_true', help='Calculate PSNR on Y channel or RGB image')

parser.add_argument("--optimizer", default="SGD", help="optimizer")
#--sgd
parser.add_argument("--momentum", type=float, default=0.9, help="learning rate")
#--adam

#data
parser.add_argument("--max_load", default=0, type=int, help="max number of samples to use; useful for reducing loading time during debugging; 0 = load all")
parser.add_argument("--style", default="Y", help="Y-channel or RGB style")
parser.add_argument("--trainset_tag", default="SR291B", help="train data directory")
parser.add_argument("--trainset_patch_size", type=int, default=96, help="train data directory")
parser.add_argument("--trainset_preload", type=int, default=0, help="train data directory")
parser.add_argument("--trainset_dir", default="/home/dataset/sr291_21x21_dn/2x/", help="train data directory")
parser.add_argument("--testset_tag", default="Set14B", help="train data directory")
parser.add_argument("--testset_dir", default="/home/dataset/set14_dnb/2x/", help="test data directory")

#model
parser.add_argument("--rgb_range", type=float, default=1.0, help="int/float images")
parser.add_argument("--scale", type=int, default=2, help="scaling factor")
parser.add_argument("--core", default="EDSR", help="core model (template specified in sr_mask_core.py)")
parser.add_argument("--checkpoint", default=None, help="checkpoint to load core from")

#eval
parser.add_argument("--eval_tag", default="psnr", help="evaluation tag; available: \"psnr, ssim\"")
parser.add_argument("--backbone_name", default="srresnet", help="backbone for get FLOPS", type=str)
# parser.add_argument("--eta", default=0.5, help="eta for balancing cost and uncertainty tradeoff", type=float)
#output
parser.add_argument("--cv_dir", default="checkpoints", help="checkpoint directory (models and logs are saved here)")
parser.add_argument("--analyze_dir", default="ANALYZE", help="Directory for analyze and visualize")
#template
parser.add_argument("--template", default=None)

# wandb
parser.add_argument("--wandb", action="store_true")
