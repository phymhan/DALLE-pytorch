import os
import sys
import ast
import math
import torch
import shutil
import random
import numpy as np
from torchvision.io import write_video
from torchvision import utils
from torch.nn import functional as F
import argparse
from pathlib import Path


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# borrowed from: https://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_args(parser, args):
    message = f"Name: {getattr(args, 'name', 'NA')}\n"
    message += '--------------- Arguments ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '------------------ End ------------------'
    # print(message)  # suppress messages to std out

    # save to the disk
    log_dir = Path(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    file_name = log_dir / 'args.txt'
    with open(file_name, 'wt') as f:
        f.write(message)
        f.write('\n')

    # save command to disk
    file_name = log_dir / 'cmd.txt'
    with open(file_name, 'wt') as f:
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            f.write('CUDA_VISIBLE_DEVICES=%s ' % os.getenv('CUDA_VISIBLE_DEVICES'))
        f.write('deepspeed ' if args.deepspeed else 'python ')
        f.write(' '.join(sys.argv))
        f.write('\n')

    # backup train code
    shutil.copyfile(sys.argv[0], log_dir / f'{os.path.basename(sys.argv[0])}.txt')


def print_models(models, args):
    if not isinstance(models, (list, tuple)):
        models = [models]
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    file_name = log_dir / 'models.txt'
    with open(file_name, 'a+') as f:
        f.write(f"Name: {getattr(args, 'name', 'NA')}\n{'-'*50}\n")
        for model in models:
            f.write(str(model))
            f.write("\n\n")


def save_image(ximg, path):
    n_sample = ximg.shape[0]
    utils.save_image(ximg, path, nrow=int(n_sample ** 0.5), normalize=True, value_range=(-1, 1))


def save_video(xseq, path):
    video = xseq.data.cpu().clamp(-1, 1)
    video = ((video+1.)/2.*255).type(torch.uint8).permute(0, 2, 3, 1)
    write_video(path, video, fps=15)


