from evaluation.psnr import calculate_psnr as psnr
from evaluation.ssim import calculate_ssim as ssim

def calculate(args, y, yt):
    if args.eval_tag == 'psnr':
        return psnr(y, yt, args.scale, args.rgb_range, args.rgb_channel)
    elif args.eval_tag == 'ssim':
        return ssim(y, yt, args.scale)
    else:
        print('[ERRO] unknown evaluation tag')
        assert(0)
        
def calculate_all(args, y, yt):
    psnr_score = psnr(y, yt, args.scale, args.rgb_range, args.rgb_channel)
    ssim_score = ssim(y, yt, args.scale)
    return (psnr_score, ssim_score)

print('[ OK ] Module "evaluation"')