import torch.nn as nn
from torch.optim import SGD
from torch import optim
from .scheduler import GradualWarmupScheduler
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                 args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        # self.opt = SGD(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.wd_reg)
        # self.opt = optim.Adam(self.net.parameters())  # weight_decay=self.args.wd_reg, lr=self.args.lr
        self._scheduler = None
        if self.args.use_lr_scheduler:
            self.set_opt()

        self.device = get_device(args)
        print(self.device)
        print(self.opt)

    def set_opt(self):
        if 'imagenet' in self.args.dataset:
            print(self.args.dataset, ' using lr_scheduler.MultiStepLR !!')
            weight_decay = 0.0005
            weight_decay = 0
            _scheduling = [30, 60, 80, 90]
            lr_decay = 0.1
            self.opt = SGD(self.net.parameters(), lr=self.args.lr, weight_decay=weight_decay)
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt,
                                                                   _scheduling,
                                                                   gamma=lr_decay)
            print(self.opt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        # print("\tIn Model: input size", x.size())
        return self.net(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass
