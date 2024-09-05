import os

def set_template(args):
    if  args.template == 'EUNAF_SRResNetxN_1est':
        print('[INFO] Template found (SRResNet SR)')
        args.style='RGB'
        args.rgb_range=1.0
        args.input_channel=3
        args.core='EUNAF_SRResNet_1est'
        args.n_resblocks = 16
        args.reduction=16
        args.n_feats=64
        args.n_estimators=3
        args.phase='test'
        args.weight = "./weights/srresnet_eunaf.t7"
        print(vars(args))
    elif  args.template == 'EUNAF_FSRCNNxN_1est':
        print('[INFO] Template found (FSRCNN SR)')
        args.style='RGB'
        args.rgb_range=1.0
        args.input_channel=3
        args.core='EUNAF_FSRCNN_1est'
        args.n_resblocks = 4
        args.reduction=16
        args.n_feats=56
        args.weight = "./weights/fsrcnn_eunaf.t7"
        print(vars(args))
    elif  args.template == 'EUNAF_CARNxN_1est':
        print('[INFO] Template found (CARN SR)')
        args.style='RGB'
        args.rgb_range=1.0
        args.input_channel=3
        args.core='EUNAF_CARN_1est'
        args.n_resblocks = 4
        args.reduction=16
        args.n_feats=64
        args.weight = "./weights/carn_eunaf.t7"
        print(vars(args))
    else:
        print('[ERRO] Template not found')
        assert(0)
