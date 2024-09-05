import torch.optim as optim

def adam_optim(all_params, args):
	return optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)