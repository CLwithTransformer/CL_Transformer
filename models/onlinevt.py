import torch
import torch.nn as nn
from torch.nn import MultiLabelSoftMarginLoss
from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
from copy import deepcopy
import numpy as np
import pdb
from datasets import get_dataset
from pytorch_metric_learning import losses as torch_losses
from apex import amp
from apex.parallel import DistributedDataParallel
from apex.parallel import convert_syncbn_model
from losses.SupConLoss import SupConLoss
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale

def get_parameter_number(net):
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return trainable_num


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Online continual learning via self-supervised Transformer')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser


class Onlinevt(ContinualModel):
    NAME = 'onlinevt'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Onlinevt, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device, args)
        self.soft_loss = MultiLabelSoftMarginLoss()
        self.dataset = get_dataset(args)
        self.total_num_class = backbone.net.num_classes
        self.gamma = None
        self.old_net = None
        self.current_task = 0
        self.iter = 0
        self.l2_current = False
        self.print_freq = 500
        self.descending = False
        self.MSE = False
        self.BCE = True
        self.use_l1_change = True
        self.class_means = None
        self.fish = None
        self.temperature = 0.07
        self.n_views = 2
        self.logsoft = nn.LogSoftmax(dim=1)
        self.MSloss = torch_losses.MultiSimilarityLoss(alpha=2, beta=10, base=0.5)
        self.ce_loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.sc_loss = SupConLoss(temperature=self.temperature)

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((self.classes_so_far, labels.to('cpu'))).unique())

        loss = self.get_loss(inputs, labels, not_aug_inputs)
        loss.backward()

        self.opt.step()

        return loss.item()

    def ncm(self, x):

        with torch.no_grad():
            self.compute_class_means()

        feats = self.net.net.contrasive_f(x)
        feats = feats.unsqueeze(1)

        pred = (self.class_means.unsqueeze(0) - feats).pow(2).sum(2)
        return -pred

    def one_hot(self, label):
        y_onehot = torch.FloatTensor(label.shape[0], self.total_num_class).to(self.device)
        y_onehot.zero_()
        y_onehot.scatter_(1, label.unsqueeze(1), 1)
        return y_onehot

    def get_loss(self, inputs, labels, not_aug_inputs):

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        if self.net.net.distill_classifier:
            outputs = self.net.net.distill_classification(inputs)
            loss += self.loss(outputs, labels) * self.args.distill_ce #/(self.current_task+1)

        if hasattr(self.args, 'ce'):
            loss = loss * self.args.ce

        if self.iter % self.print_freq == 0:
            print('current task CE loss: ', loss)

        if self.args.wd_reg:
            loss.data += self.args.wd_reg * torch.sum(self.net.get_params() ** 2)

        if self.old_net is not None and self.l2_current:
            old_output_features = self.old_net.features(inputs)
            features = self.net.features(inputs)
            loss += self.args.alpha * 0.8 * F.mse_loss(old_output_features, features)

        loss_sc = 0
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            if self.net.net.distill_classifier:
                buf_outputs = self.net.net.distill_classification(buf_inputs)
            else:
                buf_outputs = self.net(buf_inputs)
            # loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss += self.args.alpha * self.loss(buf_outputs, buf_labels)

            buf_inputs, buf_labels, _, buf_inputs_aug = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            if self.net.net.distill_classifier:
                buf_outputs = self.net.net.distill_classification(buf_inputs)
            else:
                buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

            combined_batch = torch.cat((buf_inputs, inputs))
            combined_labels = torch.cat((buf_labels, labels))
            # combined_batch = buf_inputs
            # combined_labels = buf_labels
            combined_batch_aug = self.dataset.TRANSFORM_SC(combined_batch)
            features_sc = torch.cat([self.net.net.contrasive_f(combined_batch).unsqueeze(1), self.net.net.contrasive_f(combined_batch_aug).unsqueeze(1)], dim=1)
            index = torch.randint(0, len(self.classes_so_far), (np.minimum(10, len(self.classes_so_far)),))
            focuses = self.net.net.focuses_head()[index]
            focus_labels = self.net.net.focus_labels[index]

            loss_sc = self.sc_loss(features_sc, combined_labels, focuses=focuses, focus_labels=focus_labels)

        if self.iter % self.print_freq == 0:
            print('loss_sc: ', loss_sc)
            print('total loss: ', loss)
        self.iter += 1
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)
        return loss

    def compute_class_means(self) -> None:
        """
        Computes a vector representing mean features for each class.
        """
        # This function caches class means
        transform = self.dataset.get_normalization_transform()
        class_means = []
        examples, labels, _ = self.buffer.get_all_data(transform)
        # examples, labels, _ = self.buffer.get_all_data()
        for _y in self.classes_so_far:
            x_buf = torch.stack(
                [examples[i]
                 for i in range(0, len(examples))
                 if labels[i].cpu() == _y]
            ).to(self.device)

            index = torch.randint(0, len(self.classes_so_far), (np.minimum(10, len(self.classes_so_far)),))

            focuses = self.net.net.focuses_head()[index]
            focus_labels = self.net.net.focus_labels[index]

            class_means.append(self.net.net.contrasive_f(x_buf).mean(0))

        self.class_means = torch.stack(class_means)

    def loss_trick(self, logits, labels):
        labels = labels.clone()
        ce = torch.nn.CrossEntropyLoss(reduction='mean')
        if self.params.trick['labels_trick']:
            unq_lbls = labels.unique().sort()[0]
            for lbl_idx, lbl in enumerate(unq_lbls):
                labels[labels == lbl] = lbl_idx
            # Calcualte loss only over the heads appear in the batch:
            return ce(logits[:, unq_lbls], labels)
        elif self.params.trick['separated_softmax']:
            old_ss = F.log_softmax(logits[:, self.old_labels], dim=1)
            new_ss = F.log_softmax(logits[:, self.new_labels], dim=1)
            ss = torch.cat([old_ss, new_ss], dim=1)
            for i, lbl in enumerate(labels):
                labels[i] = self.lbl_inv_map[lbl.item()]
            return F.nll_loss(ss, labels)
        elif self.params.agent in ['SCR', 'SCP']:
            SC = SupConLoss(temperature=self.temperature)
            return SC(logits, labels)
        else:
            return ce(logits, labels)

    def info_nce_loss(self, features):

        # labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.n_views)], dim=0)
        labels = torch.cat([torch.arange(features.shape[0] // 2) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def end_task(self, dataset) -> None:
        if self.args.L1 > 0:
            self.old_net = deepcopy(self.net.eval())
            self.net.train()
        self.current_task += 1

    # fill buffer according to loss
    def fill_buffer(self, mem_buffer: Buffer, dataset, t_idx: int) -> None:
        """
        Adds examples from the current task to the memory buffer
        by means of the herding strategy.
        :param mem_buffer: the memory buffer
        :param dataset: the dataset from which take the examples
        :param t_idx: the task index
        """

        ce_loss_raw = F.cross_entropy
        mode = self.net.training
        self.net.eval()
        # samples_per_class = mem_buffer.buffer_size // (self.dataset.N_CLASSES_PER_TASK * (t_idx + 1))
        samples_per_class = mem_buffer.buffer_size // len(self.classes_so_far)
        print('Classes so far:', len(self.classes_so_far))

        if t_idx > 0:
            # 1) First, subsample prior classes
            buf_x, buf_y, buf_f, buf_task_id = self.buffer.get_all_data()
            mem_buffer.empty()

            for _y in buf_y.unique():
                idx = (buf_y == _y)
                _y_x, _y_y, _y_f, _y_task_id = buf_x[idx], buf_y[idx], buf_f[idx], buf_task_id[idx]
                mem_buffer.add_data_our(
                    examples=_y_x[:samples_per_class],
                    labels=_y_y[:samples_per_class],
                    logits=_y_f[:samples_per_class],
                    task_labels=_y_task_id[:samples_per_class]
                )

        # 2) Then, fill with current tasks
        loader = dataset.not_aug_dataloader(self.args.batch_size)

        # 2.1 Extract all features
        a_x, a_y, a_logit, a_loss = [], [], [], []
        for x, y, not_norm_x in loader:
            x, y, not_norm_x = (a.to(self.device) for a in [x, y, not_norm_x])
            a_x.append(not_norm_x)
            a_y.append(y)
            outputs = self.net(x)
            a_logit.append(outputs)
            loss_raw = ce_loss_raw(outputs, y, reduction='none')
            a_loss.append(loss_raw)

        a_x, a_y, a_logit, a_loss = torch.cat(a_x), torch.cat(a_y), torch.cat(a_logit), torch.cat(a_loss)

        # 2.2 Compute class means
        for _y in a_y.unique():
            idx = (a_y == _y)
            _x, _y, _logit, _loss = a_x[idx], a_y[idx], a_logit[idx], a_loss[idx]
            _, index = _loss.sort(descending=self.descending)
            if samples_per_class < _x.shape[0]:
                index = index[:samples_per_class]

            mem_buffer.add_data_our(
                examples=_x[index].to(self.device),
                labels=_y[index].to(self.device),
                logits=_logit[index].to(self.device),
                task_labels=torch.tensor([t_idx]*len(index)).to(self.device))

        assert len(mem_buffer.examples) <= mem_buffer.buffer_size

        self.net.train(mode)

    @torch.no_grad()
    def update(self, classifier, task_size):
        old_weight_norm = torch.norm(classifier.weight[:-task_size], p=2, dim=1)
        new_weight_norm = torch.norm(classifier.weight[-task_size:], p=2, dim=1)
        self.gamma = old_weight_norm.mean() / new_weight_norm.mean()
        print(self.gamma.cpu().item())

    @torch.no_grad()
    def post_process(self, logits, task_size):
        logits[:, -task_size:] = logits[:, -task_size:] * self.gamma
        return logits

