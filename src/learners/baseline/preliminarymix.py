import torch
import time
import torch.nn as nn
import random as r
import numpy as np
import os
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix

from src.learners.baseline.base import BaseLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.models.resnet import ResNet18, ImageNet_ResNet18, ResNet18PCR, ImageNet_ResNet18PCR
from src.utils.metrics import forgetting_line
from src.utils.utils import get_device

device = get_device()


class PRELearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = Reservoir(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method,
        )
        self.iter = 0
        self.mem_iters = self.params.mem_iters
        self.irreducible_losses = torch.tensor([float("nan")] * 150000, dtype=torch.float32).to(self.device)
        self.reducible_loss = torch.tensor([float("nan")] * 150000, dtype=torch.float32).to(self.device)
        self.model_loss = torch.tensor([float("nan")] * 150000, dtype=torch.float32).to(self.device)

    def load_criterion(self):
        return SupConLoss(temperature=0.09, contrast_mode='proxy')

    def load_model(self, **kwargs):
        if self.params.dataset == 'cifar10' or self.params.dataset == 'cifar100' or self.params.dataset == 'tiny':
            return ResNet18PCR(
                dim_in=self.params.dim_in,
                nclasses=self.params.n_classes,
                nf=self.params.nf
            ).to(device)
        elif self.params.dataset == 'imagenet' or self.params.dataset == 'imagenet100':
            return ImageNet_ResNet18PCR(
                dim_in=self.params.dim_in,
                nclasses=self.params.n_classes,
                nf=self.params.nf
            ).to(device)

    def train(self, dataloader, **kwargs):
        task_name = kwargs.get('task_name', 'unknown task')
        task_id = kwargs.get('task_id', 0)
        dataloaders = kwargs.get('dataloaders', None)
        self.model = self.model.train()

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y, batch_i = batch[0], batch[1], batch[2].to(self.device)
            self.stream_idx += len(batch_x)

            for _ in range(self.mem_iters):
                mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)

                if mem_x.size(0) > 0:
                    # Combined batch
                    combined_x, combined_y = self.combine(batch_x, batch_y, mem_x, mem_y)  # (batch_size, nb_channel, img_size, img_size)
                    # Augment
                    combined_aug = self.transform_train(combined_x)
                    combined_aug1 = self.transform_1(combined_x)
                    combined_aug2 = self.transform_2(combined_aug1)
                    combined_aug3 = self.transform_3(combined_aug2)
                    combined_aug4 = self.transform_4(combined_aug3)
                    combined_augc = self.transform_cutout(combined_x)
                    combined_augj = self.transform_jigsaw(combined_x)
                    combined_augr = self.transform_rotation(combined_x)
                    combined_augw = self.transform_weak(combined_x)
                    combined_augm = self.transform_mixup(combined_x)
                    combined_augt = self.transform_translation(combined_x)
                    if self.params.augmentation == 'randaug2+rotation':
                        combined_x_aug2 = torch.cat((combined_x, combined_aug2))
                        combined_x_augr = torch.cat((combined_x, combined_augr))

                    combined_y_aug = torch.cat((combined_y, combined_y))
                    cs = combined_x.size(0)

                    # Inference
                    logits2, feas2, _ = self.model.full(combined_x_aug2)
                    logitsr, feasr, _ = self.model.full(combined_x_augr)

                    feas2_norm = torch.norm(feas2, p=2, dim=1).unsqueeze(1).expand_as(feas2)
                    feas2_normalized = feas2.div(feas2_norm + 0.000001)

                    feasr_norm = torch.norm(feasr, p=2, dim=1).unsqueeze(1).expand_as(feasr)
                    feasr_normalized = feasr.div(feasr_norm + 0.000001)

                    feas_aug = self.model.pcrLinear.L.weight[combined_y_aug.long()]
                    feas_aug_norm = torch.norm(feas_aug, p=2, dim=1).unsqueeze(1).expand_as(feas_aug)
                    feas_aug_normalized = feas_aug.div(feas_aug_norm + 0.000001)

                    cos_features2 = torch.cat([feas2_normalized.unsqueeze(1), feas_aug_normalized.unsqueeze(1)], dim=1)
                    cos_featuresr = torch.cat([feasr_normalized.unsqueeze(1), feas_aug_normalized.unsqueeze(1)], dim=1)

                    # Loss
                    loss = (self.criterion(cos_features2, combined_y_aug.long()) + self.criterion(cos_featuresr, combined_y_aug.long()))
                    self.loss = loss.item()
                    print(f"Loss {self.loss:.3f}  batch {j}", end="\r")
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    self.iter += 1

            # Update reservoir buffer
            self.buffer.update(imgs=batch_x, labels=batch_y, model=self.model)

            if (j == (len(dataloader) - 1)) and (j > 0):
                print(f"Task : {task_name}   batch {j}/{len(dataloader)}   Loss : {loss.item():.4f}    time : {time.time() - self.start:.4f}s")

    def evaluate_and_save_irreducible_loss(self, dataloader, **kwargs):
        with torch.no_grad():
            self.model.eval()
            print("Computing irreducible loss full training dataset.")
            i = 0
            start_index = 0
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                for sample in dataloader:
                    inputs = sample[0]
                    labels = sample[1]
                    indexes = sample[2]

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    indexes = indexes.to(device)

                    combined_x_aug1 = self.transform_1(inputs)
                    combined_x_aug2 = self.transform_2(combined_x_aug1)
                    combined_x_aug3 = self.transform_3(combined_x_aug2)
                    combined_x_aug4 = self.transform_4(combined_x_aug3)
                    combined_x_augc = self.transform_cutout(inputs)
                    combined_x_augj = self.transform_jigsaw(inputs)
                    combined_x_augr = self.transform_rotation(inputs)
                    if self.params.augmentation == 'randaug2+rotation':
                        logits2 = self.model.logits(combined_x_aug2)
                        logitsr = self.model.logits(combined_x_augr)
                    loss = (nn.functional.cross_entropy(logits2, labels.long(), reduction='none') + nn.functional.cross_entropy(logitsr, labels.long(), reduction='none'))
                    self.irreducible_losses[indexes] = loss


    def evaluate_and_save_model_loss(self, dataloader, **kwargs):
        with torch.no_grad():
            self.model.eval()
            print("Computing irreducible loss full training dataset.")
            i = 0
            start_index = 0
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                for sample in dataloader:
                    inputs = sample[0]
                    labels = sample[1]
                    indexes = sample[2]

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    indexes = indexes.to(device)

                    combined_x_aug1 = self.transform_1(inputs)
                    combined_x_aug2 = self.transform_2(combined_x_aug1)
                    combined_x_aug3 = self.transform_3(combined_x_aug2)
                    combined_x_aug4 = self.transform_4(combined_x_aug3)
                    combined_x_augc = self.transform_cutout(inputs)
                    combined_x_augj = self.transform_jigsaw(inputs)
                    combined_x_augr = self.transform_rotation(inputs)

                    if self.params.augmentation == 'randaug2+rotation':
                        logits2 = self.model.logits(combined_x_aug2)
                        logitsr = self.model.logits(combined_x_augr)
                    loss = (nn.functional.cross_entropy(logits2, labels.long(), reduction='none') + nn.functional.cross_entropy(logitsr, labels.long(), reduction='none'))
                    self.model_loss[indexes] = loss
                    self.reducible_loss[indexes] = self.model_loss[indexes] - self.irreducible_losses[indexes]


    def print_results(self, task_id):
        n_dashes = 20
        pad_size = 8
        print('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)

        print('-' * n_dashes + "ACCURACY" + '-' * n_dashes)
        for line in self.results:
            print('Acc'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line),
                  f"{np.nanmean(line):.4f}")

    def combine(self, batch_x, batch_y, mem_x, mem_y):
        mem_x, mem_y = mem_x.to(self.device), mem_y.to(self.device)
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        combined_x = torch.cat([mem_x, batch_x])
        combined_y = torch.cat([mem_y, batch_y])
        if self.params.memory_only:
            return mem_x, mem_y
        else:
            return combined_x, combined_y
