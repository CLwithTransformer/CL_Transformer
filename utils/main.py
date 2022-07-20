# coding=UTF-8
import sys
import os
sys.path.append(os.getcwd())
import importlib

_cpath_='/usr/local/python/3.7.7/lib/python3.7/site-packages'
if _cpath_ in sys.path:
    sys.path.remove(_cpath_)
from datasets import NAMES as DATASET_NAMES
sys.path.insert(50, _cpath_)

from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import ContinualDataset
from utils.continual_training import train as ctrain
from datasets import get_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed
import megengine as mge
# from torchsummary import summary
from torchsummaryX import summary
import torch
from thop import profile
from torchstat import stat
import torchvision
from ptflops import get_model_complexity_info
from apex.parallel import DistributedDataParallel as DDP_apex
from apex.parallel import convert_syncbn_model
import torch.distributed as dist
from confusion_matrix import plot_confusion
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(random.choice([1]))

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

def main():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    parser.add_argument("--local_rank", type=int, default=0, help='node rank for distributed training')
    parser.add_argument("--num_workers", default=4, type=int)
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=DATASET_NAMES,
                            help='Which dataset to perform experiments on.')
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
        if hasattr(args, 'buffer_size'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        for key, value in best.items():
            setattr(args, key, value)
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    setattr(args, 'GAN', 'GAN')
    setattr(args, 'use_albumentations', False)
    setattr(args, 'use_apex', False)
    setattr(args, 'use_distributed', True)
    if 'imagenet' in args.dataset:
        setattr(args, 'use_lr_scheduler', True)
    else:
        setattr(args, 'use_lr_scheduler', False)
    if torch.cuda.device_count() <= 1 or args.dataset == 'seq-mnist':
        setattr(args, 'use_distributed', False)

    if args.model == 'mer': setattr(args, 'batch_size', 1)
    dataset = get_dataset(args)
    if args.model == 'our' or args.model == 'our_reservoir':
        backbone = dataset.get_backbone_our()
    elif args.model == 'onlinevt':
        backbone = dataset.get_backbone_cct()
    else:
        backbone = dataset.get_backbone()

    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    if args.use_apex or args.use_distributed:
        dist.init_process_group(backend='nccl')  # , init_method='env://'
        torch.cuda.set_device(args.local_rank)
        model.to(model.device)

        if args.use_apex:
            model = convert_syncbn_model(model)
            model.net, model.opt = amp.initialize(model.net, model.opt, opt_level='O1')
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        print("Let's use", torch.cuda.device_count(), "GPUs!!!")
        if hasattr(model.net, 'net') and hasattr(model.net.net, 'distill_classifier'):
            distill_classifier = model.net.net.distill_classifier
            distill_classification = model.net.net.distill_classification
            update_gamma = model.net.net.update_gamma
            if args.use_apex:
                print("Let's use apex !!!")
                model.net.net = DDP_apex(model.net.net, delay_allreduce=True)  # , device_ids=[args.local_rank]
            else:
                model.net.net = DDP(model.net.net, device_ids=[args.local_rank], output_device=args.local_rank,
                                    broadcast_buffers=False, find_unused_parameters=False)
            setattr(model.net.net, 'distill_classifier', distill_classifier)
            setattr(model.net.net, 'distill_classification', distill_classification)
            setattr(model.net.net, 'update_gamma', update_gamma)
        else:
            get_params = model.net.get_params
            model.net = DDP(model.net, device_ids=[args.local_rank], output_device=args.local_rank,
                            broadcast_buffers=False)  # , device_ids=[args.local_rank]
            setattr(model.net, 'get_params', get_params)
            pass
    else:
        model.to(model.device)

    print(args)
    if hasattr(model, 'loss_name'):
        print('loss name:  ', model.loss_name)

    if isinstance(dataset, ContinualDataset):
        train(model, dataset, args)
    else:
        assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
        ctrain(args)

if __name__ == '__main__':
    main()





