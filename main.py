import numpy as np
import torch, argparse
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

    if args.mode == 'train' : net.train()
    elif args.mode == 'test' : net.test()
    elif args.mode == 'generate' : net.generate(num_sample = args.batch_size,
                                                target = args.target,
                                                epsilon = args.epsilon,
                                                iteration = args.iteration)
    else : return

    print('[*] Finished')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='toynet template')
    parser.add_argument('--epoch', default = 20, type=int, help='epoch size')
    parser.add_argument('--batch_size', default = 100, type=int, help='mini-batch size')
    parser.add_argument('--lr', default = 2e-4, type=float, help='learning rate')
    parser.add_argument('--y_dim', default = 10, type=int, help='the number of classes')
    parser.add_argument('--target', default = -1, type=int, help='target class for targeted generation')
    parser.add_argument('--eps', default = 1e-9, type=float, help='epsilon')
    parser.add_argument('--env_name', default='main', type=str, help='experiment name')
    parser.add_argument('--dataset', default='FMNIST', type=str, help='dataset type')
    parser.add_argument('--dset_dir', default='datasets', type=str, help='dataset directory path')
    parser.add_argument('--summary_dir', default='summary', type=str, help='summary directory path')
    parser.add_argument('--output_dir', default='output', type=str, help='output directory path')
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory path')
    parser.add_argument('--load_ckpt', type=str, default='', help='')
    parser.add_argument('--cuda', type=str2bool, default=True, help='enable cuda')
    parser.add_argument('--silent', type=str2bool, default=False, help='')
    parser.add_argument('--mode', type=str, default='train', help='train / test / generate')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--iteration', default=1, type=int, help='the number of iteration for FGSM')
    parser.add_argument('--epsilon', default=0.03, type=float, help='epsilon for FGSM')
    parser.add_argument('--tensorboard', default=False, type=str2bool, help='enable tensorboard')
    parser.add_argument('--visdom', default=False, type=str2bool, help='enable visdom')
    parser.add_argument('--visdom_port', default=55558, type=str, help='visdom port')
    args = parser.parse_args()

    main(args)
