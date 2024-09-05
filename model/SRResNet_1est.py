import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common

import numpy as np



def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.ModuleList(layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        common.initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class MSRResNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, args, conv=common.default_conv):
        super(MSRResNet, self).__init__()
        
        self.in_nc = in_nc = args.input_channel
        self.out_nc = out_nc = args.input_channel 
        self.nf = nf = args.n_feats 
        self.upscale = scale = args.scale[0] if type(args.scale)==tuple else args.scale    # [HxW] or [HxH]
        self.nb = nb = args.n_resblocks
        self.n_estimators = n_estimators = args.n_estimators
        self.kernel_size = kernel_size = 3

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.recon_trunk = make_layer(basic_block, nb)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        common.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last], 0.1)
        if self.upscale == 4:
            common.initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        for i in range(self.nb):
            out = self.recon_trunk(fea)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out
    
class Estimator(nn.Module):
    def __init__(self):
        super(Estimator, self).__init__()
        self.lastOut = nn.Linear(32, 4)

        # Condtion network
        self.CondNet = nn.Sequential(nn.Conv2d(3, 128, 4, 4), nn.LeakyReLU(0.1),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1),
                                     nn.Conv2d(128, 32, 1))
        common.initialize_weights([self.CondNet], 0.1)
    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AvgPool2d(out.size()[2])(out)
        out = out.view(out.size(0), -1)
        out = self.lastOut(out)
        return out
    
class EUNAF_MSRResNet_1est(MSRResNet):
    def __init__(self, args, conv=common.default_conv):
        super(EUNAF_MSRResNet_1est, self).__init__(args, conv=conv) 
        self.n_estimators = min(args.n_estimators, self.nb // 2)

        self.predictors = self.init_intermediate_out(self.n_estimators-1, conv, out_channels=args.input_channel, last_act=False)
        self.estimator = Estimator()
        self.cost_dict = torch.tensor([0, 2.04698, 3.66264, 5.194]) / 5.194 # norm cost
        self.counts = [0, 0, 0, 0]
            
    def get_n_estimators(self):
        return self.n_estimators
        
    def init_intermediate_out(self, num_blocks, conv,
                              out_channels=1, is_estimator=False, 
                              last_act=False):
        
        interm_predictors = nn.ModuleList()
        
        for i in range(num_blocks):
            if is_estimator:
                return self.estimator
            else:
                
                if i==num_blocks-1:
                    m_tail = [
                        conv(self.nf, self.nf*4, 3),
                        nn.PixelShuffle(2), nn.LeakyReLU(0.1),
                        conv(self.nf, self.nf*2, 3),
                        nn.PixelShuffle(2), nn.LeakyReLU(0.1),
                        conv(self.nf//2, self.nf//2, 3), nn.LeakyReLU(0.1),
                        conv(self.nf//2, out_channels, 3)
                    ] 
                else:
                    m_tail = [
                        conv(self.nf, self.nf*4, 3),
                        nn.PixelShuffle(2), nn.LeakyReLU(0.1),
                        conv(self.nf, self.nf, 3),
                        nn.PixelShuffle(2), nn.LeakyReLU(0.1),
                        conv(self.nf//4, self.nf//4, 3), nn.LeakyReLU(0.1),
                        conv(self.nf//4, out_channels, 1)
                    ]       
            common.initialize_weights(m_tail, 0.1)
            if last_act: m_tail.append(nn.ELU())
            interm_predictors.append(nn.Sequential(*m_tail))
            
        return interm_predictors
    
    def freeze_backbone(self):
        for n, p in self.named_parameters():
            if 'estimator' not in n:
                p.requires_grad = False
            if p.requires_grad:
                print(n, end=' ')
    
    def enable_estimators_only(self):
        for n, p in self.named_parameters():
            if 'estimator' not in n:
                p.requires_grad = False
            if p.requires_grad:
                print(n, end=' ')
                
    def forward_backbone(self, x):
        fea = self.lrelu(self.conv_first(x))
        for i in range(self.nb):
            out = self.recon_trunk[i](fea)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        
        outs = [None, None, None, out]
        masks = None
        
        return outs, masks
    
    def forward(self, x):
        
        masks = self.estimator(x)   # Bx3
        
        outs = list()
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        outs.append(
            F.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        )
        
        fea = self.lrelu(self.conv_first(x))
        # gap_range = np.arange(2, self.nb, self.gap)
        # tmp_gap_range = self.gap_range[:-1] if len(gap_range)==self.n_estimators else self.gap_range
        tmp_gap_range = [self.nb-1, self.nb-1]
        
        cnt = 0
        for i in range(self.nb):
            fea = self.recon_trunk[i](fea)
            
            if i == 8:  # ee 1
                tmp_out = self.predictors[0](fea)
                outs.append(tmp_out + base)
            
            if i == self.nb-1:  # ee 2
                tmp_out = self.predictors[1](fea)
                outs.append(tmp_out+base)
                        
        if self.upscale == 4:
            fea = self.lrelu(self.pixel_shuffle(self.upconv1(fea)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(fea)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(fea)))
        out = self.conv_last(self.lrelu(self.HRconv(out)))  
        
        out += base
        outs.append(out)
                        
        return outs, masks
    
    def eunaf_forward(self, x):
        
        masks = self.estimator(x)   # Bx3
        
        outs = list()
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        outs.append(
            F.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        )
        
        fea = self.lrelu(self.conv_first(x))
        # gap_range = np.arange(2, self.nb, self.gap)
        # tmp_gap_range = self.gap_range[:-1] if len(gap_range)==self.n_estimators else self.gap_range
        tmp_gap_range = [self.nb-1, self.nb-1]
        
        cnt = 0
        for i in range(self.nb):
            fea = self.recon_trunk[i](fea)
            
            if i == 8:  # ee 1
                tmp_out = self.predictors[0](fea)
                outs.append(tmp_out + base)
            
            if i == self.nb-1:  # ee 2
                tmp_out = self.predictors[1](fea)
                outs.append(tmp_out+base)
                        
        if self.upscale == 4:
            fea = self.lrelu(self.pixel_shuffle(self.upconv1(fea)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(fea)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(fea)))
        out = self.conv_last(self.lrelu(self.HRconv(out)))  
        
        out += base
        outs.append(out)
                        
        return outs, masks
    
    def eunaf_infer(self, x, eta=0.0, imscore=None):
        assert x.shape[0]==1, "only 1 patch at a time"
        masks = self.estimator(x)   # Bx4
        # norm_masks = masks - torch.amin(masks, dim=1) / (torch.amax(masks, dim=1) - torch.amin(masks, dim=1))
        norm_masks = masks
        
        imscores = np.array(imscore) # N
        q1 = 10
        p0 = (imscores <= q1).astype(int)
        blank_vector = torch.zeros_like(masks)
        blank_vector[:, 0] = torch.tensor(p0)
        
        path_decision = masks + eta*self.cost_dict.to(x.device) - 1.0 * blank_vector.to(x.device)
        decision = torch.argmin(path_decision).int().item()
        self.counts[decision] += 1
        
        outs = list()
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        
        if decision==0:
            return F.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        
        
        fea = self.lrelu(self.conv_first(x))
        tmp_gap_range = [self.nb-1, self.nb-1]
        
        cnt = 0
        for i in range(self.nb):
            fea = self.recon_trunk[i](fea)
            
            if i == 8:  # ee 1
                tmp_out = self.predictors[0](fea)
                # outs.append(tmp_out + base)
                if decision==1: return tmp_out+base
            
            if i == self.nb-1:  # ee 2
                tmp_out = self.predictors[1](fea)
                # outs.append(tmp_out+base)
                if decision==2: return tmp_out + base
                
                        
        if self.upscale == 4:
            fea = self.lrelu(self.pixel_shuffle(self.upconv1(fea)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(fea)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(fea)))
        out = self.conv_last(self.lrelu(self.HRconv(out)))  
        
        out += base
        # outs.append(out)
                        
        # return outs, masks
        return out 