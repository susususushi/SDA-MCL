import copy

import random
import kornia.augmentation
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import random as r
import numpy as np
import os
import pandas as pd
import wandb
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale, RandomInvert

from src.learners.ccldc.baseccl import BaseCCLLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.models.resnet import ResNet18, ResNet18PCR, ImageNet_ResNet18, ImageNet_ResNet18PCR, cosLinear
from src.utils.metrics import forgetting_line
from src.utils.utils import get_device
from src.utils.augment import MixUpAugment
from copy import deepcopy

device = get_device()


class SDAMCLLearner(BaseCCLLearner):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = Reservoir(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method,
        )
        self.kd_lambda = self.params.kd_lambda
        self.iter = 0

    def load_model(self, **kwargs):
        if self.params.dataset == 'cifar10' or self.params.dataset == 'cifar100':
            return ResNet18PCR(
                dim_in=self.params.dim_in,
                nclasses=self.params.n_classes,
                nf=self.params.nf
            ).to(device)
        elif self.params.dataset == 'tiny':
            model = ResNet18PCR(dim_in=self.params.dim_in, nclasses=self.params.n_classes, nf=self.params.nf)
            model.pcrLinear = cosLinear(self.params.dim_in, self.params.n_classes)
            return model.to(device)
        elif self.params.dataset == 'imagenet' or self.params.dataset == 'imagenet100':
            return ImageNet_ResNet18PCR(
                dim_in=self.params.dim_in,
                nclasses=self.params.n_classes,
                nf=self.params.nf
            ).to(device)

    def load_criterion(self):
        return SupConLoss(temperature=0.09, contrast_mode='proxy')

    def train(self, dataloader, **kwargs):
        task_name = kwargs.get('task_name', 'unknown task')
        task_id = kwargs.get('task_id', 0)
        dataloaders = kwargs.get('dataloaders', None)
        self.model1 = self.model1.train()
        self.model2 = self.model2.train()
        self.all_labels = torch.tensor(self.params.labels_order[:])

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)

            for _ in range(self.params.mem_iters):

                mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)

                if mem_x.size(0) > 0:
                    combined_x, combined_y = self.combine(batch_x, batch_y, mem_x, mem_y)
                    # Augment
                    bs = combined_x.size(0)
                    combined_aug = self.transform_train(combined_x)

                    combined_x_aug = torch.cat((combined_x, combined_aug))
                    combined_y_aug = torch.cat((combined_y, combined_y))

                    logits1, feas1 = self.model1.full(combined_x_aug)
                    logits2, feas2 = self.model2.full(combined_x_aug)

                    feas1_norm = torch.norm(feas1, p=2, dim=1).unsqueeze(1).expand_as(feas1)
                    feas1_normalized = feas1.div(feas1_norm + 0.000001)

                    feas1_aug = self.model1.pcrLinear.L.weight[combined_y_aug.long()]
                    feas1_aug_norm = torch.norm(feas1_aug, p=2, dim=1).unsqueeze(1).expand_as(feas1_aug)
                    feas1_aug_normalized = feas1_aug.div(feas1_aug_norm + 0.000001)
                    cos1_features = torch.cat([feas1_normalized.unsqueeze(1), feas1_aug_normalized.unsqueeze(1)], dim=1)

                    feas2_norm = torch.norm(feas2, p=2, dim=1).unsqueeze(1).expand_as(feas2)
                    feas2_normalized = feas2.div(feas2_norm + 0.000001)

                    feas2_aug = self.model2.pcrLinear.L.weight[combined_y_aug.long()]
                    feas2_aug_norm = torch.norm(feas2_aug, p=2, dim=1).unsqueeze(1).expand_as(feas2_aug)
                    feas2_aug_normalized = feas2_aug.div(feas2_aug_norm + 0.000001)
                    cos2_features = torch.cat([feas2_normalized.unsqueeze(1), feas2_aug_normalized.unsqueeze(1)], dim=1)

                    loss_pscl = self.criterion(features=cos1_features, labels=combined_y_aug.long())
                    loss2_pscl = self.criterion(features=cos2_features, labels=combined_y_aug.long())

                    features1 = torch.cat([feas1_normalized[:bs, ].unsqueeze(1), feas2_normalized[bs:, ].detach().unsqueeze(1)], dim=1)
                    features2 = torch.cat([feas2_normalized[:bs, ].unsqueeze(1), feas1_normalized[bs:, ].detach().unsqueeze(1)], dim=1)

                    MCL = SupConLoss(temperature=0.07, contrast_mode='all')
                    loss_mcl = MCL(features1, combined_y.long())
                    loss2_mcl = MCL(features2, combined_y.long())

                    combined_x, combined_y = self.combine(batch_x, batch_y, mem_x, mem_y)
                    combined_aug1 = self.transform_1(combined_x)
                    combined_aug2 = self.transform_2(combined_aug1)
                    combined_aug3 = self.transform_3(combined_aug2)
                    combined_aug4 = self.transform_4(combined_aug3)
                    # self.plt_img(combined_x)
                    # self.plt_img(combined_aug)
                    # self.plt_img(combined_aug1)
                    # self.plt_img(combined_aug2)
                    # self.plt_img(combined_aug3)

                    logits1_randaug2, feas1_randaug2 = self.model1.full(combined_aug2)
                    logits2_randaug2, feas2_randaug2 = self.model2.full(combined_aug2)

                    feas1_randaug2_norm = torch.norm(feas1_randaug2, p=2, dim=1).unsqueeze(1).expand_as(feas1_randaug2)
                    feas1_randaug2_normalized = feas1_randaug2.div(feas1_randaug2_norm + 0.000001)

                    feas1_randaug2_aug = self.model1.pcrLinear.L.weight[combined_y.long()]
                    feas1_randaug2_aug_norm = torch.norm(feas1_randaug2_aug, p=2, dim=1).unsqueeze(1).expand_as(feas1_randaug2_aug)
                    feas1_randaug2_aug_normalized = feas1_randaug2_aug.div(feas1_randaug2_aug_norm + 0.000001)
                    cos1_randaug2_features = torch.cat([feas1_randaug2_normalized.unsqueeze(1), feas1_randaug2_aug_normalized.unsqueeze(1)], dim=1)

                    feas2_randaug2_norm = torch.norm(feas2_randaug2, p=2, dim=1).unsqueeze(1).expand_as(feas2_randaug2)
                    feas2_randaug2_normalized = feas2_randaug2.div(feas2_randaug2_norm + 0.000001)

                    feas2_randaug2_aug = self.model2.pcrLinear.L.weight[combined_y.long()]
                    feas2_randaug2_aug_norm = torch.norm(feas2_randaug2_aug, p=2, dim=1).unsqueeze(1).expand_as(feas2_randaug2_aug)
                    feas2_randaug2_aug_normalized = feas2_randaug2_aug.div(feas2_randaug2_aug_norm + 0.000001)
                    cos2_randaug2_features = torch.cat([feas2_randaug2_normalized.unsqueeze(1), feas2_randaug2_aug_normalized.unsqueeze(1)], dim=1)

                    loss_randaug2_pscl = self.criterion(features=cos1_randaug2_features, labels=combined_y.long())
                    loss2_randaug2_pscl = self.criterion(features=cos2_randaug2_features, labels=combined_y.long())

                    features1 = torch.cat([feas1_normalized[:bs, ].unsqueeze(1), feas2_randaug2_normalized.detach().unsqueeze(1)], dim=1)
                    features2 = torch.cat([feas2_normalized[:bs, ].unsqueeze(1), feas1_randaug2_normalized.detach().unsqueeze(1)], dim=1)

                    loss_randaug2_mcl = MCL(features1, combined_y.long())
                    loss2_randaug2_mcl = MCL(features2, combined_y.long())

                    loss = 0.5 * (loss_pscl + loss_mcl) + self.kl_loss(logits1, logits2.detach()) \
                           + self.params.randaug2_weight * (loss_randaug2_pscl + loss_randaug2_mcl) + self.kl_loss(logits1_randaug2, logits2_randaug2.detach())

                    loss2 = 0.5 * (loss2_pscl + loss2_mcl) + self.kl_loss(logits2, logits1.detach()) \
                            + self.params.randaug2_weight * (loss2_randaug2_pscl + loss2_randaug2_mcl) + self.kl_loss(logits2_randaug2, logits1_randaug2.detach())

                    batch_rotation = self.transform_rotation(combined_x.to(self.device))
                    label_rotation = combined_y.to(self.device)
                    logits1_rotation, feas1_rotation = self.model1.full(batch_rotation)
                    logits2_rotation, feas2_rotation = self.model2.full(batch_rotation)

                    feas1_rotation_norm = torch.norm(feas1_rotation, p=2, dim=1).unsqueeze(1).expand_as(feas1_rotation)
                    feas1_rotation_normalized = feas1_rotation.div(feas1_rotation_norm + 0.000001)

                    feas1_rotation_aug = self.model1.pcrLinear.L.weight[label_rotation.long()]
                    feas1_rotation_aug_norm = torch.norm(feas1_rotation_aug, p=2, dim=1).unsqueeze(1).expand_as(
                        feas1_rotation_aug)
                    feas1_rotation_aug_normalized = feas1_rotation_aug.div(feas1_rotation_aug_norm + 0.000001)
                    cos1_rotation_features = torch.cat([feas1_rotation_normalized.unsqueeze(1), feas1_rotation_aug_normalized.unsqueeze(1)], dim=1)

                    feas2_rotation_norm = torch.norm(feas2_rotation, p=2, dim=1).unsqueeze(1).expand_as(feas2_rotation)
                    feas2_rotation_normalized = feas2_rotation.div(feas2_rotation_norm + 0.000001)

                    feas2_rotation_aug = self.model2.pcrLinear.L.weight[label_rotation.long()]
                    feas2_rotation_aug_norm = torch.norm(feas2_rotation_aug, p=2, dim=1).unsqueeze(1).expand_as(feas2_rotation_aug)
                    feas2_rotation_aug_normalized = feas2_rotation_aug.div(feas2_rotation_aug_norm + 0.000001)

                    cos2_rotation_features = torch.cat([feas2_rotation_normalized.unsqueeze(1), feas2_rotation_aug_normalized.unsqueeze(1)], dim=1)

                    loss_rotation_pscl = self.criterion(features=cos1_rotation_features, labels=label_rotation.long())
                    loss2_rotation_pscl = self.criterion(features=cos2_rotation_features, labels=label_rotation.long())

                    features1_rotation = torch.cat([feas1_normalized[:bs, ].unsqueeze(1), feas2_rotation_normalized.detach().unsqueeze(1)], dim=1)
                    features2_rotation = torch.cat([feas2_normalized[:bs, ].unsqueeze(1), feas1_rotation_normalized.detach().unsqueeze(1)], dim=1)

                    loss_rotation_mcl = MCL(features1_rotation, label_rotation.long())
                    loss2_rotation_mcl = MCL(features2_rotation, label_rotation.long())

                    loss = loss + self.params.rotation_weight * (loss_rotation_pscl + loss_rotation_mcl) + self.kl_loss(logits1_rotation, logits2_rotation.detach())
                    loss2 = loss2 + self.params.rotation_weight * (loss2_rotation_pscl + loss2_rotation_mcl) + self.kl_loss(logits2_rotation, logits1_rotation.detach())
                    self.loss = loss.item()
                    self.loss2 = loss2.item()
                    print(f"Loss (Peer1) : {loss.item():.4f}  Loss (Peer2) : {loss2.item():.4f}   batch {j}", end="\r")
                    self.optim1.zero_grad()
                    loss.backward()
                    self.optim1.step()

                    self.optim2.zero_grad()
                    loss2.backward()
                    self.optim2.step()

                    self.iter += 1
            # Update reservoir buffer
            self.buffer.update(imgs=batch_x, labels=batch_y)

            if (j == (len(dataloader) - 1)) and (j > 0):
                print(
                    f"Task : {task_name}   batch {j}/{len(dataloader)}   Loss (Peer1) : {loss.item():.4f}  Loss (Peer2) : {loss2.item():.4f}   time : {time.time() - self.start:.4f}s"
                )

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

    def plt_img(self, combined_x_mix):
        num_samples = combined_x_mix.shape[0]
        height = combined_x_mix.shape[2]
        width = combined_x_mix.shape[3]
        num_samples_per_row = 4
        num_rows = -(-num_samples // num_samples_per_row)

        fig, axs = plt.subplots(num_rows, num_samples_per_row, figsize=(16, num_rows * 4))

        for i in range(num_samples):
            row = i // num_samples_per_row
            col = i % num_samples_per_row

            sample_image = combined_x_mix[i].cpu().numpy().transpose(1, 2, 0)
            sample_image = np.clip(sample_image, 0, 1)

            axs[row, col].imshow(sample_image)
            axs[row, col].axis('off')
            axs[row, col].set_title(f'Sample {i + 1}')

        for i in range(num_samples, num_rows * num_samples_per_row):
            row = i // num_samples_per_row
            col = i % num_samples_per_row
            fig.delaxes(axs[row, col])

        plt.tight_layout()
        plt.show()
