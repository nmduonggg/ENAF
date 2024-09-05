import numpy as np
import torch
from loss.l1 import l1_loss
from loss.l2 import l2_loss

#general method to be called
def create_loss_func(tag):
    if   tag == 'L1':
        return l1_loss()
    elif tag == 'L2':
        return l2_loss()
    else:
        print('[ERRO] unknown loss tag')
        assert(0)

print('[ OK ] Module "loss"')