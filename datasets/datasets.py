"""datasets.py"""
import os

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class UnknownDatasetError(Exception):
    def __str__(self):
        return "unknown datasets error"


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    if 'MNIST' in name:
        root = os.path.join(dset_dir, 'MNIST')
        train_kwargs = {'root':root, 'train':True, 'transform':transform, 'download':True}
        test_kwargs = {'root':root, 'train':False, 'transform':transform, 'download':False}
        dset = MNIST

    else:
        raise UnknownDatasetError()

    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=1,
                              pin_memory=True,
                              drop_last=True)

    test_data = dset(**test_kwargs)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=1,
                             pin_memory=True,
                             drop_last=False)

    data_loader = dict()
    data_loader['train'] = train_loader
    data_loader['test'] = test_loader

    return data_loader


if __name__ == '__main__':
    import argparse
    os.chdir('..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--dset_dir', type=str, default='datasets')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    data_loader = return_data(args)
