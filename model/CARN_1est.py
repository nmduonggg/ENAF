import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from model import common
import numpy as np

class EResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out
    
class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, stride=stride),
            nn.ReLU(True)
        ]
        super(BasicBlock, self).__init__(*m)
        
class UpsampleBlock(nn.Module):
    def __init__(self,
                 n_channels, scale, multi_scale,
                 group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale=4):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self,
                 n_channels, scale,
                 group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        return out

class Block(nn.Module):
    def __init__(self, nf,
                 group=1):
        super(Block, self).__init__()

        self.b1 = EResidualBlock(nf, nf, group=group)
        self.c1 = BasicBlock(nf*2, nf, 1, 1, 0)
        self.c2 = BasicBlock(nf*3, nf, 1, 1, 0)
        self.c3 = BasicBlock(nf*4, nf, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3
        
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

class CARN_M(nn.Module):
    def __init__(self, args):
        super(CARN_M, self).__init__()
        self.in_nc = in_nc = args.input_channel
        self.out_nc = out_nc = args.input_channel
        self.nf = nf = args.n_feats
        self.scale = scale = args.scale[0] if type(args.scale)==tuple else args.scale
        multi_scale=False
        
        self.group = group = 4

        self.rgb_range = rgb_range = args.rgb_range # = 1.0
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.entry = nn.Conv2d(in_nc, nf, 3, 1, 1)

        self.b1 = Block(nf, group=group)
        self.b2 = Block(nf, group=group)
        self.b3 = Block(nf, group=group)
        self.c1 = BasicBlock(nf*2, nf, 1, 1, 0)
        self.c2 = BasicBlock(nf*3, nf, 1, 1, 0)
        self.c3 = BasicBlock(nf*4, nf, 1, 1, 0)
        
        self.upsample = UpsampleBlock(nf, scale=scale, 
                                          multi_scale=multi_scale,
                                          group=group)
        self.exit = nn.Conv2d(nf, out_nc, 3, 1, 1)
                
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=self.scale)

        out = self.exit(out)
        out = self.add_mean(out)

        return out
    
class EUNAF_CARN_1est(CARN_M):
    def __init__(self, args, conv=common.default_conv):
        super(EUNAF_CARN_1est, self).__init__(args)
        self.n_estimators = 3
        self.predictors = self.init_intermediate_out(self.n_estimators-1, conv, out_channels=args.input_channel)
        self.estimator = Estimator()
        
        self.cost_dict = torch.tensor([0, 778.55, 868.86, 1161.72]) / 1161.72
        self.counts = [0, 0, 0, 0]
        
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
                        nn.Conv2d(self.nf, 32*4, 3, 1, 1, groups=self.group), nn.LeakyReLU(0.1),
                        nn.PixelShuffle(2),
                        nn.Conv2d(32, 32*4, 3, 1, 1, groups=self.group), nn.LeakyReLU(0.1),
                        nn.PixelShuffle(2), 
                        nn.Conv2d(32, out_channels, 3, 1, 1)
                    ] 
                else:
                    m_tail = [
                        nn.Conv2d(self.nf, 16*4, 3, 1, 1, groups=self.group), nn.LeakyReLU(0.1),
                        nn.PixelShuffle(2),
                        nn.Conv2d(16, 16*4, 3, 1, 1, groups=self.group), nn.LeakyReLU(0.1),
                        nn.PixelShuffle(2), 
                        nn.Conv2d(16, out_channels, 3, 1, 1)
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
                
    def forward_backbone(self, x):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=self.scale)

        out = self.exit(out)
        out = self.add_mean(out)
        
        outs = [None, None, None, out]
        masks = None

        return outs, masks
                
    def forward(self, x):
        
        masks = self.estimator(x)
        ee0 = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x
        
        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        
        ee1 = self.predictors[0](o3)
        ee1 = self.add_mean(ee1)
        
        ee2 = self.predictors[1](o3)
        ee2 = self.add_mean(ee2)

        out = self.upsample(o3, scale=self.scale)

        out = self.exit(out)
        out = self.add_mean(out)
        
        outs = [ee0, ee1, ee2, out]

        return outs, masks
                
    def eunaf_forward(self, x):
        
        masks = self.estimator(x)
        ee0 = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x
        
        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        
        ee1 = self.predictors[0](o3)
        ee1 = self.add_mean(ee1)
        
        ee2 = self.predictors[1](o3)
        ee2 = self.add_mean(ee2)

        out = self.upsample(o3, scale=self.scale)

        out = self.exit(out)
        out = self.add_mean(out)
        
        outs = [ee0, ee1, ee2, out]

        return outs, masks
    
    def eunaf_infer(self, x, eta=0.0, imscore=None):
        
        masks = self.estimator(x)
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
        
        if decision==0:
            return F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x
        
        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        
        if decision==1:
            ee1 = self.predictors[0](o3)
            ee1 = self.add_mean(ee1)
            
        if decision==2:
            ee2 = self.predictors[1](o3)
            ee2 = self.add_mean(ee2)

        out = self.upsample(o3, scale=self.scale)

        out = self.exit(out)
        out = self.add_mean(out)
        
        # outs = [ee0, ee1, ee2, out]

        # return outs, masks
        return out