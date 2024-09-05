import torch.optim as optim

def sgd_optim(all_params, args):
	return optim.SGD(all_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)