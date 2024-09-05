from .SRResNet import EUNAF_MSRResNet
from .FSRCNN_1est import EUNAF_FSRCNN_1est
from .CARN import EUNAF_CARN
from .SRResNet_1est import EUNAF_MSRResNet_1est
from .CARN_1est import EUNAF_CARN_1est


def config(args):
    arch = args.core.split("-")
    name = arch[0]
    if name=='EUNAF_SRResNet':
        return EUNAF_MSRResNet(args)
    elif name=='EUNAF_CARN':
        return EUNAF_CARN(args)
    elif name=='EUNAF_FSRCNN_1est':
        return EUNAF_FSRCNN_1est(args)
    elif name=='EUNAF_SRResNet_1est':
        return EUNAF_MSRResNet_1est(args)
    elif name=='EUNAF_CARN_1est':
        return EUNAF_CARN_1est(args)
    
    
    else:
        assert(0), 'No configuration found'