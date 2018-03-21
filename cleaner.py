"""cleaner.py"""

import argparse
from pathlib import Path

from utils.utils import rm_dir


def clean(args):
    """Remove directories relevant to specified experiment name given as env_name"""

    env_name = args.env_name

    ckpt_dir = Path(args.ckpt_dir).joinpath(env_name)
    summary_dir = Path(args.summary_dir).joinpath(env_name)
    output_dir = Path(args.output_dir).joinpath(env_name)

    rm_dir(ckpt_dir)
    rm_dir(summary_dir)
    rm_dir(output_dir)

    print('[*] Cleaning Finished ! ')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--summary_dir', type=str, default='summary')
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()

    clean(args)
