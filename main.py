"""main.py"""
import argparse

import numpy as np
import torch

from solver import Solver
from utils.utils import str2bool

def main(args):

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    print()
    print('[ARGUMENTS]')
    print(args)
    print()

    net = Solver(args)

    if args.mode == 'train':
        net.train()
    elif args.mode == 'test':
        net.test()
    elif args.mode == 'generate':
        net.generate(num_sample=args.batch_size,
                     target=args.target,
                     epsilon=args.epsilon,
                     alpha=args.alpha,
                     iteration=args.iteration)
    elif args.mode == 'universal':
        net.universal(args)
    else: return

    print('[*] Finished')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='toynet template')
    parser.add_argument('--epoch', type=int, default=20, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=100, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--y_dim', type=int, default=10, help='the number of classes')
    parser.add_argument('--target', type=int, default=-1, help='target class for targeted generation')
    parser.add_argument('--eps', type=float, default=1e-9, help='epsilon')
    parser.add_argument('--env_name', type=str, default='main', help='experiment name')
    parser.add_argument('--dataset', type=str, default='FMNIST', help='dataset type')
    parser.add_argument('--dset_dir', type=str, default='datasets', help='dataset directory path')
    parser.add_argument('--summary_dir', type=str, default='summary', help='summary directory path')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory path')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='checkpoint directory path')
    parser.add_argument('--load_ckpt', type=str, default='', help='')
    parser.add_argument('--cuda', type=str2bool, default=True, help='enable cuda')
    parser.add_argument('--silent', type=str2bool, default=False, help='')
    parser.add_argument('--mode', type=str, default='train', help='train / test / generate / universal')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--iteration', type=int, default=1, help='the number of iteration for FGSM')
    parser.add_argument('--epsilon', type=float, default=0.03, help='epsilon for FGSM and i-FGSM')
    parser.add_argument('--alpha', type=float, default=2/255, help='alpha for i-FGSM')
    parser.add_argument('--tensorboard', type=str2bool, default=False, help='enable tensorboard')
    parser.add_argument('--visdom', type=str2bool, default=False, help='enable visdom')
    parser.add_argument('--visdom_port', type=str, default=55558, help='visdom port')
    args = parser.parse_args()

    main(args)
