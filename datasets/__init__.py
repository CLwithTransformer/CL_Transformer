from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace
from datasets.seq_cifar100 import SequentialCIFAR100



NAMES = {
    SequentialCIFAR100.NAME: SequentialCIFAR100,
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)


def get_gcl_dataset(args: Namespace):
    """
    Creates and returns a GCL dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    return NAMES[args.dataset](args)
