from optimizer.sgd import sgd_optim
from optimizer.adam import adam_optim

def create_optimizer(all_params, args):
	algo = args.optimizer
	if algo == 'SGD':
		return sgd_optim(all_params, args)
	elif algo == 'Adam':
		return adam_optim(all_params, args)
	else:
		print('[ERRO] unknown optimizer')
		assert(0)

print('[ OK ] Module "optimizer"')