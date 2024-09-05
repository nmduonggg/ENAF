import torch.nn as nn
from model import common
import torch
import numpy as np

import functools
import torch.nn as nn
import torch.nn.functional as F
import torch


class FSRCNN_net(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(FSRCNN_net, self).__init__()
        
        self.input_channels = input_channels = args.input_channel
        self.upscale = upscale = args.scale[0] if type(args.scale)==tuple else args.scale    # [HxW] or [HxH]
        self.nf = nf = args.n_feats
        self.s = s = 12
        self.m = m = args.n_resblocks
        
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=nf, kernel_size=5, stride=1, padding=2),
            nn.PReLU())

        self.layers = []
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=nf, out_channels=s, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))
        for _ in range(m):
            self.layers.append(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=nf, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))

        self.body_conv = torch.nn.ModuleList(self.layers)

        # Deconvolution
        self.tail_conv = nn.ConvTranspose2d(in_channels=nf, out_channels=input_channels, kernel_size=9,
                                            stride=upscale, padding=3, output_padding=1)


        common.initialize_weights([self.head_conv, self.body_conv, self.tail_conv], 0.1)

    def forward(self, x):
        fea = self.head_conv(x)
        fea = self.body_conv(fea)
        out = self.tail_conv(fea)
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
    
class EUNAF_FSRCNN_1est(FSRCNN_net):
    def __init__(self, args, conv=common.default_conv):
        super(EUNAF_FSRCNN_1est, self).__init__(args, conv=conv)
        self.n_estimators = 3
        self.predictors = self.init_intermediate_out(self.n_estimators-1, conv, out_channels=args.input_channel)
        self.estimator = Estimator()
        
        self.cost_dict = torch.tensor([0, 146.42, 315.45, 468.2]) / 468.2 # norm cost
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
                        conv(self.s, 36, 3), nn.LeakyReLU(0.1),
                        nn.ConvTranspose2d(in_channels=36, out_channels=self.input_channels, kernel_size=9,
                                            stride=self.upscale, padding=3, output_padding=1)
                    ] 
                else:
                    m_tail = [
                        conv(self.s, 16, 3), nn.LeakyReLU(0.1),
                        nn.ConvTranspose2d(in_channels=16, out_channels=self.input_channels, kernel_size=9,
                                            stride=self.upscale, padding=3, output_padding=1)
                    ]       
            common.initialize_weights(m_tail, 0.1)
            if last_act: m_tail.append(nn.ELU())
            interm_predictors.append(nn.Sequential(*m_tail))
            
        return interm_predictors
    
    def freeze_backbone(self):
        for n, p in self.named_parameters():
            if 'estimator' not in n:
                p.requires_grad = False
            else:
                print(n, end="; ")
                
    def unfreeze_predictors(self):
        for n, p in self.named_parameters():
            if 'estimator' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
                print(n, end="; ")
                
    def forward_backbone(self, x):
        fea = self.head_conv(x)
        for i, b in enumerate(self.body_conv):
            fea = self.body_conv[i](fea)
        out = self.tail_conv(fea)
        
        outs = [None, None, None, out]
        masks = None
        
        return outs, masks

    def forward(self, x):
        masks = self.estimator(x)
        
        fea = self.head_conv(x)
        outs = [
            F.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        ]
        
        for i, b in enumerate(self.body_conv):
            fea = self.body_conv[i](fea)
            
            if i==2:
                tmp_out = self.predictors[0](fea)
                outs.append(tmp_out)
            
            if i==len(self.body_conv)-2: 
                tmp_out = self.predictors[1](fea)
                outs.append(tmp_out)
        
        out = self.tail_conv(fea)
        outs.append(out)
        
        return outs, masks
    
    def eunaf_forward(self, x):
        masks = self.estimator(x)
        
        fea = self.head_conv(x)
        outs = [
            F.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        ]
        
        for i, b in enumerate(self.body_conv):
            fea = self.body_conv[i](fea)
            
            if i==2:
                tmp_out = self.predictors[0](fea)
                outs.append(tmp_out)
            
            if i==len(self.body_conv)-2: 
                tmp_out = self.predictors[1](fea)
                outs.append(tmp_out)
        
        out = self.tail_conv(fea)
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
        
        fea = self.head_conv(x)
        if decision == 0:
            return F.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        
        for i, b in enumerate(self.body_conv):
            fea = self.body_conv[i](fea)
            
            if i==2 and decision==1:
                tmp_out = self.predictors[0](fea)
                # outs.append(tmp_out)
                return tmp_out
            
            if (i==len(self.body_conv)-2) and decision==2: 
                tmp_out = self.predictors[1](fea)
                # outs.append(tmp_out)
                return tmp_out
        
        out = self.tail_conv(fea)
        # outs.append(out)
        
        return out