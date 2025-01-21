from random import random
import random
import torch
import logging as lg
import os
import pickle
import time
import os
import datetime as dt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import random as r
import json
import torchvision
import wandb
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.manifold import TSNE
import seaborn as sns
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from torchvision import transforms
import torchvision.transforms.functional as TF
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale, RandomInvert
from copy import deepcopy
from torchvision.transforms import RandAugment
from umap import UMAP
from torch.distributions import Categorical
from PIL import Image
from src.utils.utils import save_model
from src.models import resnet
from src.utils.metrics import forgetting_line
from src.utils.utils import get_device
from src.utils.augment import MixUpAugment, CutMixAugment, SaliencyMixAugment, JFMixAugment, StyleAugment

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
device = get_device()


class RandomLargeDegreeRotation(nn.Module):
    def __init__(self):
        super(RandomLargeDegreeRotation, self).__init__()

    def forward(self, img):
        angle = random.choice([-90, 90, 180])
        return transforms.functional.rotate(img, angle)
class JigsawTransform(nn.Module):
    def __init__(self, n=3):
        super(JigsawTransform, self).__init__()
        self.n = n  # n x n jigsaw puzzle

    def forward(self, imgs):
        return torch.stack([self.jigsaw_transform(img) for img in imgs]).to(imgs.device)

    def jigsaw_transform(self, img):
        img = transforms.ToPILImage()(img)

        w, h = img.size
        s = w // self.n  # Size of each block
        blocks = []

        for i in range(self.n):
            for j in range(self.n):
                block = img.crop((j * s, i * s, (j + 1) * s, (i + 1) * s))
                blocks.append(block)

        random.shuffle(blocks)

        new_img = Image.new('RGB', (w, h))
        for i in range(self.n):
            for j in range(self.n):
                new_img.paste(blocks[i * self.n + j], (j * s, i * s))

        return transforms.ToTensor()(new_img)

class CutoutTransform(nn.Module):
    def __init__(self, n_holes=1, length=16):
        super(CutoutTransform, self).__init__()
        self.n_holes = n_holes
        self.length = length

    def forward(self, imgs):
        return torch.stack([self.cutout_transform(img) for img in imgs]).to(imgs.device)

    def cutout_transform(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask).expand_as(img).to(img.device)
        img = img * mask

        return img

class BaseCCLLearner(torch.nn.Module):
    def __init__(self, args):
        """Learner abstract class
        Args:
            args: argparser arguments
        """
        super().__init__()
        self.params = args
        self.multiplier = self.params.multiplier
        self.device = get_device()
        self.init_tag()
        self.model1 = self.load_model()
        # print(self.model1.state_dict())
        self.model2 = self.load_model()
        # print(self.model2.state_dict())
        self.optim1 = self.load_optim1()
        self.optim2 = self.load_optim2()
        self.buffer = None
        self.start = time.time()
        self.criterion = self.load_criterion()
        if self.params.tensorboard:
            self.writer = self.load_writer()
        self.classifiers_list = ['ncm']  # Classifiers used for evaluating representation
        self.loss = 0
        self.stream_idx = 0
        self.results = []
        self.results_clustering = []
        self.results_1 = []
        self.results_2 = []
        self.results_clustering_forgetting = []
        self.results_forgetting = []
        self.results_forgetting_1 = []
        self.results_forgetting_2 = []

        normalize = nn.Identity()
        if self.params.tf_type == 'partial':
            self.transform_train = nn.Sequential(
                torchvision.transforms.RandomCrop(self.params.img_size, padding=4),
                RandomHorizontalFlip(),
                normalize
            ).to(device)
        elif self.params.tf_type == 'moderate':
            self.transform_train = nn.Sequential(
                RandomResizedCrop(size=(self.params.img_size, self.params.img_size), scale=(0.6, 1.)),
                RandomHorizontalFlip(),
                RandomGrayscale(p=0.2),
                normalize
            ).to(device)
        else:
            self.transform_train = nn.Sequential(
                RandomResizedCrop(size=(self.params.img_size, self.params.img_size), scale=(self.params.min_crop, 1.)),
                RandomHorizontalFlip(),
                ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                RandomGrayscale(p=0.2),
                normalize
            ).to(device)

        self.transform_rotation = nn.Sequential(
            RandomLargeDegreeRotation()
        ).to(device)

        self.transform_jigsaw = nn.Sequential(
            JigsawTransform(n=2)
        ).to(device)

        self.transform_cutout = nn.Sequential(
            CutoutTransform(n_holes=1, length=112)
        ).to(device)

        self.transform_1 = nn.Sequential(
            torchvision.transforms.ConvertImageDtype(torch.uint8),
            RandAugment(self.params.randaug_n, self.params.randaug_m),
            torchvision.transforms.ConvertImageDtype(torch.float32),
            normalize
        ).to(device)

        self.transform_2 = nn.Sequential(
            torchvision.transforms.ConvertImageDtype(torch.uint8),
            RandAugment(self.params.randaug_n, self.params.randaug_m),
            torchvision.transforms.ConvertImageDtype(torch.float32),
            normalize
        ).to(device)

        self.transform_3 = nn.Sequential(
            torchvision.transforms.ConvertImageDtype(torch.uint8),
            RandAugment(self.params.randaug_n, self.params.randaug_m),
            torchvision.transforms.ConvertImageDtype(torch.float32),
            normalize
        ).to(device)

        self.transform_4 = nn.Sequential(
            torchvision.transforms.ConvertImageDtype(torch.uint8),
            RandAugment(self.params.randaug_n, self.params.randaug_m),
            torchvision.transforms.ConvertImageDtype(torch.float32),
            normalize
        ).to(device)

        self.transform_test = nn.Sequential(
            normalize
        ).to(device)

    def init_tag(self):
        """Initialise tag for experiment
        """
        if self.params.training_type == 'inc':
            self.params.tag = f"{self.params.learner},{self.params.dataset},m{self.params.mem_size}mbs{self.params.mem_batch_size}sbs{self.params.batch_size}{self.params.tag}"
        elif self.params.training_type == "blurry":
            self.params.tag = \
                f"{self.params.learner},{self.params.dataset},m{self.params.mem_size}mbs{self.params.mem_batch_size}sbs{self.params.batch_size}blurry{self.params.blurry_scale}{self.params.tag}"
        else:
            self.params.tag = f"{self.params.learner},{self.params.dataset},{self.params.epochs}b{self.params.batch_size},uni{self.params.tag}"
        print(f"Using the following tag for this experiment : {self.params.tag}")

    def load_writer(self):
        """Initialize tensorboard summary writer
        """
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(self.params.tb_root, self.params.tag))
        return writer

    def save(self, path):
        lg.debug("Saving checkpoint...")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        torch.save(self.model1.state_dict(), os.path.join(path, 'model1.pth'))
        torch.save(self.model2.state_dict(), os.path.join(path, 'model2.pth'))

        # with open(os.path.join(path, f"memory.pkl"), 'wb') as memory_file:
        #     pickle.dump(self.buffer, memory_file)

    def resume(self, path):
        self.model1.load_state_dict(torch.load(os.path.join(path, 'model1.pth')))
        self.model2.load_state_dict(torch.load(os.path.join(path, 'model2.pth')))
        # with open(os.path.join(path, f"memory.pkl"), 'rb') as f:
        #     self.buffer = pickle.load(f)
        # f.close()
        torch.cuda.empty_cache()

    def load_model(self):
        """Load model
        Returns:
            untrained torch backbone model
        """
        return NotImplementedError

    def load_optim1(self):
        """Load optimizer for training
        Returns:
            torch.optim: torch optimizer
        """
        if self.params.optim == 'Adam':
            optimizer = torch.optim.Adam(self.model1.parameters(), lr=self.params.learning_rate,
                                         weight_decay=self.params.weight_decay)
        elif self.params.optim == 'AdamW':
            optimizer = torch.optim.AdamW(self.model1.parameters(), lr=self.params.learning_rate,
                                          weight_decay=self.params.weight_decay)
        elif self.params.optim == 'SGD':
            optimizer = torch.optim.SGD(
                self.model1.parameters(),
                lr=self.params.learning_rate,
                momentum=self.params.momentum,
                weight_decay=self.params.weight_decay
            )
        else:
            raise Warning('Invalid optimizer selected.')
        return optimizer

    def load_optim2(self):
        """Load optimizer for training
        Returns:
            torch.optim: torch optimizer
        """
        lr = self.params.learning_rate
        if self.params.optim == 'Adam':
            optimizer = torch.optim.Adam(self.model2.parameters(), lr=lr, weight_decay=self.params.weight_decay)
        elif self.params.optim == 'AdamW':
            optimizer = torch.optim.AdamW(self.model2.parameters(), lr=lr, weight_decay=self.params.weight_decay)
        elif self.params.optim == 'SGD':
            optimizer = torch.optim.SGD(
                self.model2.parameters(),
                lr=lr,
                momentum=self.params.momentum,
                weight_decay=self.params.weight_decay
            )
        else:
            raise Warning('Invalid optimizer selected.')
        return optimizer

    def load_scheduler(self):
        raise NotImplementedError

    def load_criterion(self):
        raise NotImplementedError

    def train(self, dataloader, task, **kwargs):
        raise NotImplementedError

    def evaluate(self, dataloaders, task_id, **kwargs):
        with torch.no_grad():
            self.model1.eval()
            self.model2.eval()
            accs_ens = []
            accs_1 = []
            accs_2 = []
            preds_ens = []
            preds_1 = []
            preds_2 = []
            all_targets = []
            tag = ''
            for j in range(task_id + 1):
                test_preds_ens, test_preds_1, test_preds_2, test_targets = self.encode_logits(dataloaders[f"test{j}"])
                acc_ens = accuracy_score(test_targets, test_preds_ens)
                acc_1 = accuracy_score(test_targets, test_preds_1)
                acc_2 = accuracy_score(test_targets, test_preds_2)

                accs_ens.append(acc_ens)
                accs_1.append(acc_1)
                accs_2.append(acc_2)
                # Wandb logs
                if not self.params.no_wandb:
                    preds_ens = np.concatenate([preds_ens, test_preds_ens])
                    preds_1 = np.concatenate([preds_1, test_preds_1])
                    preds_2 = np.concatenate([preds_2, test_preds_2])
                    all_targets = np.concatenate([all_targets, test_targets])
                    wandb.log({
                        tag + f"ens_acc_{j}": acc_ens,
                        "task_id": task_id
                    })
                    wandb.log({
                        tag + f"net1_acc_{j}": acc_1,
                        "task_id": task_id
                    })
                    wandb.log({
                        tag + f"net2_acc_{j}": acc_2,
                        "task_id": task_id
                    })

            # Make confusion matrix
            if not self.params.no_wandb:
                # re-index to have classes in task order
                all_targets = [self.params.labels_order.index(int(i)) for i in all_targets]
                preds_ens = [self.params.labels_order.index(int(i)) for i in preds_ens]
                preds_1 = [self.params.labels_order.index(int(i)) for i in preds_1]
                preds_2 = [self.params.labels_order.index(int(i)) for i in preds_2]
                cm_ens = np.log(1 + confusion_matrix(all_targets, preds_ens))
                cm_1 = np.log(1 + confusion_matrix(all_targets, preds_1))
                cm_2 = np.log(1 + confusion_matrix(all_targets, preds_2))
                fig = plt.matshow(cm_ens)
                wandb.log({
                    tag + f"ens_cm": fig,
                    "task_id": task_id
                })
                fig = plt.matshow(cm_1)
                wandb.log({
                    tag + f"net1_cm": fig,
                    "task_id": task_id
                })
                fig = plt.matshow(cm_2)
                wandb.log({
                    tag + f"net2_cm": fig,
                    "task_id": task_id
                })

            for _ in range(self.params.n_tasks - task_id - 1):
                accs_ens.append(np.nan)
                accs_1.append(np.nan)
                accs_2.append(np.nan)

            self.results.append(accs_ens)
            self.results_1.append(accs_1)
            self.results_2.append(accs_2)

            line = forgetting_line(pd.DataFrame(self.results), task_id=task_id, n_tasks=self.params.n_tasks)
            line = line[0].to_numpy().tolist()
            self.results_forgetting.append(line)

            line = forgetting_line(pd.DataFrame(self.results_1), task_id=task_id, n_tasks=self.params.n_tasks)
            line = line[0].to_numpy().tolist()
            self.results_forgetting_1.append(line)

            line = forgetting_line(pd.DataFrame(self.results_2), task_id=task_id, n_tasks=self.params.n_tasks)
            line = line[0].to_numpy().tolist()
            self.results_forgetting_2.append(line)

            self.print_results(task_id)

            return np.nanmean(self.results[-1]), np.nanmean(self.results_forgetting[-1]), np.nanmean(
                self.results_1[-1]), np.nanmean(self.results_forgetting_1[-1]), np.nanmean(
                self.results_2[-1]), np.nanmean(self.results_forgetting_2[-1])

    def evaluate_offline(self, dataloaders, epoch=0):
        with torch.no_grad():
            self.model1.eval()
            self.model2.eval()
            test_preds_ens, test_preds_1, test_preds_2, test_targets = self.encode_logits(dataloaders['test'])
            acc_ens = accuracy_score(test_targets, test_preds_ens)
            acc_1 = accuracy_score(test_targets, test_preds_1)
            acc_2 = accuracy_score(test_targets, test_preds_2)
            self.results_1.append(acc_1)
            self.results_2.append(acc_2)
            self.results.append(acc_ens)

        print(f"ACCURACY {self.results[-1]}")
        return self.results[-1], self.results_1[-1], self.results_2[-1]

    def evaluate_clustering(self, dataloaders, task_id, **kwargs):
        try:
            results = self._evaluate_clustering(dataloaders, task_id, **kwargs)
        except:
            results = 0, 0, 0, 0, 0, 0
        return results

    def _evaluate_clustering(self, dataloaders, task_id, **kwargs):
        with torch.no_grad():
            self.model1.eval()
            self.model2.eval()

            # Train classifier on labeled data
            step_size = int(self.params.n_classes / self.params.n_tasks)
            mem_representations_ens, mem_representations_n1, mem_representations_n2, mem_labels = self.get_mem_rep_labels_ens(
                use_proj=self.params.eval_proj)

            # UMAP visualization
            # reduction = self.umap_reduction(mem_representations_ens.cpu().numpy())
            # plt.figure()
            # figure = plt.scatter(reduction[:, 0], reduction[:, 1], c=mem_labels, cmap='Spectral', s=1)
            # if not self.params.no_wandb:
            #     wandb.log({
            #         "ens_umap": wandb.Image(figure),
            #         "task_id": task_id
            #     })
            classifiers_ens = self.init_classifiers()
            classifiers_n1 = self.init_classifiers()
            classifiers_n2 = self.init_classifiers()
            classifiers_ens = self.fit_classifiers(classifiers=classifiers_ens, representations=mem_representations_ens,
                                                   labels=mem_labels)
            classifiers_n1 = self.fit_classifiers(classifiers=classifiers_n1, representations=mem_representations_n1,
                                                  labels=mem_labels)
            classifiers_n2 = self.fit_classifiers(classifiers=classifiers_n2, representations=mem_representations_n2,
                                                  labels=mem_labels)

            accs = []
            accs1 = []
            accs2 = []
            representations_ens = {}
            representations_n1 = {}
            representations_n2 = {}
            targets_ens = {}
            targets_n1 = {}
            targets_n2 = {}
            preds_ens = []
            preds_1 = []
            preds_2 = []
            all_targets = []
            tag = 'stu'

            for j in range(task_id + 1):
                test_representation, test_representation_n1, test_representation_n2, test_targets = self.encode_fea(
                    dataloaders[f"test{j}"])
                representations_ens[f"test{j}"] = test_representation
                targets_ens[f"test{j}"] = test_targets
                representations_n1[f"test{j}"] = test_representation_n1
                targets_n1[f"test{j}"] = test_targets
                representations_n2[f"test{j}"] = test_representation_n2
                targets_n2[f"test{j}"] = test_targets

                test_preds_ens = classifiers_ens[0].predict(representations_ens[f'test{j}'])
                test_preds_1 = classifiers_n1[0].predict(representations_n1[f'test{j}'])
                test_preds_2 = classifiers_n2[0].predict(representations_n2[f'test{j}'])

                acc_ens = accuracy_score(targets_ens[f"test{j}"], test_preds_ens)
                acc_1 = accuracy_score(targets_n1[f"test{j}"], test_preds_1)
                acc_2 = accuracy_score(targets_n2[f"test{j}"], test_preds_2)

                accs.append(acc_ens)
                accs1.append(acc_1)
                accs2.append(acc_2)
                # Wandb logs
                if not self.params.no_wandb:
                    preds_ens = np.concatenate([preds_ens, test_preds_ens])
                    preds_1 = np.concatenate([preds_1, test_preds_1])
                    preds_2 = np.concatenate([preds_2, test_preds_2])
                    all_targets = np.concatenate([all_targets, test_targets])
                    wandb.log({
                        tag + f"ncm_ens_acc_{j}": acc_ens,
                        "task_id": task_id
                    })
                    wandb.log({
                        tag + f"ncm_net1_acc_{j}": acc_1,
                        "task_id": task_id
                    })
                    wandb.log({
                        tag + f"ncm_net2_acc_{j}": acc_2,
                        "task_id": task_id
                    })

            # Make confusion matrix
            if not self.params.no_wandb:
                # re-index to have classes in task order
                all_targets = [self.params.labels_order.index(int(i)) for i in all_targets]
                preds_ens = [self.params.labels_order.index(int(i)) for i in preds_ens]
                preds_1 = [self.params.labels_order.index(int(i)) for i in preds_1]
                preds_2 = [self.params.labels_order.index(int(i)) for i in preds_2]
                cm_ens = np.log(1 + confusion_matrix(all_targets, preds_ens))
                cm_1 = np.log(1 + confusion_matrix(all_targets, preds_1))
                cm_2 = np.log(1 + confusion_matrix(all_targets, preds_2))
                fig = plt.matshow(cm_ens)
                wandb.log({
                    tag + f"ncm_ens_cm": fig,
                    "task_id": task_id
                })
                fig = plt.matshow(cm_1)
                wandb.log({
                    tag + f"ncm_net1_cm": fig,
                    "task_id": task_id
                })
                fig = plt.matshow(cm_2)
                wandb.log({
                    tag + f"ncm_net2_cm": fig,
                    "task_id": task_id
                })

            for _ in range(self.params.n_tasks - task_id - 1):
                accs.append(np.nan)
                accs1.append(np.nan)
                accs2.append(np.nan)

            self.results_clustering.append(accs)
            self.results_1.append(accs1)
            self.results_2.append(accs2)

            line = forgetting_line(pd.DataFrame(self.results_clustering), task_id=task_id, n_tasks=self.params.n_tasks)
            line = line[0].to_numpy().tolist()
            self.results_clustering_forgetting.append(line)

            line = forgetting_line(pd.DataFrame(self.results_1), task_id=task_id, n_tasks=self.params.n_tasks)
            line = line[0].to_numpy().tolist()
            self.results_forgetting_1.append(line)

            line = forgetting_line(pd.DataFrame(self.results_2), task_id=task_id, n_tasks=self.params.n_tasks)
            line = line[0].to_numpy().tolist()
            self.results_forgetting_2.append(line)

            return np.nanmean(self.results_clustering[-1]), np.nanmean(
                self.results_clustering_forgetting[-1]), np.nanmean(self.results_1[-1]), np.nanmean(
                self.results_forgetting_1[-1]), np.nanmean(self.results_2[-1]), np.nanmean(
                self.results_forgetting_2[-1])

    def encode_fea(self, dataloader, nbatches=-1):
        i = 0
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]

                inputs = inputs.to(device)
                feat1 = self.model1(self.transform_test(inputs))
                feat2 = self.model2(self.transform_test(inputs))

                feat_ens = (feat1 + feat2) / 2.0

                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat1 = feat1.cpu().numpy()
                    all_feat2 = feat2.cpu().numpy()
                    all_feat_ens = feat_ens.cpu().numpy()
                    i += 1
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat_ens = np.vstack([all_feat_ens, feat_ens.cpu().numpy()])
                    all_feat1 = np.vstack([all_feat1, feat1.cpu().numpy()])
                    all_feat2 = np.vstack([all_feat2, feat2.cpu().numpy()])

        return all_feat_ens, all_feat1, all_feat2, all_labels

    def encode_logits(self, dataloader, nbatches=-1):
        i = 0
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for sample in dataloader:
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]

                inputs = inputs.to(device)
                feat1 = self.model1.logits(self.transform_test(inputs))
                feat2 = self.model2.logits(self.transform_test(inputs))
                feat_ens = (feat1 + feat2) / 2.0

                preds_ens = feat_ens.argmax(dim=1)
                preds_1 = feat1.argmax(dim=1)
                preds_2 = feat2.argmax(dim=1)

                if i == 0:
                    all_labels = labels.cpu().numpy()
                    all_feat1 = preds_1.cpu().numpy()
                    all_feat2 = preds_2.cpu().numpy()
                    all_feat_ens = preds_ens.cpu().numpy()
                    i += 1
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat_ens = np.hstack([all_feat_ens, preds_ens.cpu().numpy()])
                    all_feat1 = np.hstack([all_feat1, preds_1.cpu().numpy()])
                    all_feat2 = np.hstack([all_feat2, preds_2.cpu().numpy()])
        return all_feat_ens, all_feat1, all_feat2, all_labels

    def init_classifiers(self):
        """Initiliaze every classifier for representation transfer learning
        Returns:
            List of initialized classifiers
        """
        # logreg = LogisticRegression(random_state=self.params.seed)
        # knn = KNeighborsClassifier(3)
        # linear = MLPClassifier(hidden_layer_sizes=(200), activation='identity', max_iter=500, random_state=self.params.seed)
        # svm = SVC()
        ncm = NearestCentroid()
        # return [logreg, knn, linear, svm, ncm]
        return [ncm]

    @ignore_warnings(category=ConvergenceWarning)
    def fit_classifiers(self, classifiers, representations, labels):
        """Fit every classifiers on representation - labels pairs
        Args:
            classifiers : List of sklearn classifiers
            representations (torch.Tensor): data representations 
            labels (torch.Tensor): data labels
        Returns:
            List of trained classifiers
        """
        for clf in classifiers:
            clf.fit(representations.cpu(), labels)
        return classifiers

    def after_eval(self, **kwargs):
        pass

    def before_eval(self, **kwargs):
        pass

    def train_inc(self, **kwargs):
        raise NotImplementedError

    def train_blurry(self, **kwargs):
        raise NotImplementedError

    def get_mem_rep_labels_ens(self, eval=True, use_proj=False):
        """Compute every representation -labels pairs from memory
        Args:
            eval (bool, optional): Whether to turn the mdoel in evaluation mode. Defaults to True.
        Returns:
            representation - labels pairs
        """
        if eval:
            self.model1.eval()
            self.model2.eval()
        mem_imgs, mem_labels = self.buffer.get_all()
        batch_s = 10
        n_batch = len(mem_imgs) // batch_s
        all_reps = []
        n1_reps = []
        n2_reps = []
        for i in range(n_batch):
            mem_imgs_b = mem_imgs[i * batch_s:(i + 1) * batch_s].to(self.device)
            mem_imgs_b = self.transform_test(mem_imgs_b)
            mem_representations_b1 = self.model1(mem_imgs_b)
            mem_representations_b2 = self.model2(mem_imgs_b)
            mem_representations_b = (mem_representations_b1 + mem_representations_b2) / 2.0
            n1_reps.append(mem_representations_b1)
            n2_reps.append(mem_representations_b2)
            all_reps.append(mem_representations_b)
        mem_representations_ens = torch.cat(all_reps, dim=0)
        mem_representations_n1 = torch.cat(n1_reps, dim=0)
        mem_representations_n2 = torch.cat(n2_reps, dim=0)
        return mem_representations_ens, mem_representations_n1, mem_representations_n2, mem_labels

    def kl_loss(self, logits_stu, logits_tea, temperature=4.0):
        """
        Args:
            logits_stu: student logits
            logits_tea: teacher logits
            temperature: temperature
        Returns:
            distillation loss
        """
        pred_teacher = F.softmax(logits_tea / temperature, dim=1)
        log_pred_student = F.log_softmax(logits_stu / temperature, dim=1)
        loss_kd = F.kl_div(
            log_pred_student,
            pred_teacher,
            reduction='none'
        ).sum(1).mean(0) * (temperature ** 2)
        return loss_kd

    def save_results(self, table_path, run_id):
        with open(table_path + '/run_time = {}.txt'.format(run_id), 'w') as f:
            for each_task in self.results:
                f.write(str(list(each_task)) + "\n")
            mean_acc = np.array(self.results)[-1].mean()
            f.write('last task avr acc: ' + str(mean_acc) + '\n')

    def save_ncmresults(self, table_path, run_id):
        with open(table_path + '/run_time = {}.txt'.format(run_id), 'w') as f:
            for each_task in self.results_clustering:
                f.write(str(list(each_task)) + "\n")
            mean_acc = np.array(self.results_clustering)[-1].mean()
            f.write('last task avr acc: ' + str(mean_acc) + '\n')

    def umap_reduction(self, representation):
        umap_reducer = UMAP()
        umap_result = umap_reducer.fit_transform(representation)
        return umap_result

    def print_results(self, task_id):
        n_dashes = 20
        pad_size = 8
        print('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)

        print('-' * n_dashes + "ACCURACY" + '-' * n_dashes)
        for line in self.results:
            print('Acc'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line),
                  f"{np.nanmean(line):.4f}")

    def backward_transfer(self):
        n_tasks = len(self.results)
        bt = 0
        for i in range(1, n_tasks):
            for j in range(i):
                bt += self.results[i][j] - self.results[j][j]

        return bt / (n_tasks * (n_tasks - 1) / 2)

    def learning_accuracy(self):
        n_tasks = len(self.results)
        la = 0
        for i in range(n_tasks):
            la += self.results[i][i]
        return la / n_tasks

    def reletive_forgetting(self):
        n_tasks = len(self.results)
        rf = 0
        max = np.nanmax(np.array(self.results), axis=0)
        for i in range(n_tasks - 1):
            if max[i] != 0:
                rf += self.results_forgetting[-1][i] / max[i]
            else:
                rf += 1

        return rf / n_tasks

    def get_entropy(self, dataloaders, task_id):
        trainloader = dataloaders[f"train{task_id}"]
        testloader = dataloaders[f"test{task_id}"]

        train_ce = 0
        train_en = 0
        test_ce = 0
        test_en = 0
        samples = 0

        self.model1.eval()
        self.model2.eval()

        for i, batch in enumerate(trainloader):
            inputs = batch[0].to(device)
            labels = batch[1].to(device).long()
            samples += inputs.shape[0]
            outputs = self.model1.logits(self.transform_test(inputs))
            prob = torch.softmax(outputs, dim=1)
            train_ce += torch.nn.CrossEntropyLoss(reduction='sum')(outputs, labels).item()
            train_en += Categorical(probs=prob).entropy().sum().item()

        train_ce /= samples
        train_en /= samples

        samples = 0

        for i, batch in enumerate(testloader):
            inputs = batch[0].to(device)
            labels = batch[1].to(device).long()
            samples += inputs.shape[0]
            outputs = self.model1.logits(self.transform_test(inputs))
            prob = torch.softmax(outputs, dim=1)
            test_ce += torch.nn.CrossEntropyLoss(reduction='sum')(outputs, labels).item()
            test_en += Categorical(probs=prob).entropy().sum().item()

        test_ce /= samples
        test_en /= samples

        self.model1.train()
        self.model2.train()
        return train_ce, train_en, test_ce, test_en
    def load_in100_class_names(self, file_path):
        class_to_name = {}
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                class_to_name[parts[0]] = parts[1]
        return class_to_name

    def cam_visualization(self, dataloaders, task_id):
        self.model1.eval()
        for i in [3]:
            class_to_name = self.load_in100_class_names('./In100en.txt')
            class_to_idx = dataloaders[f"test{i}"].dataset.class_to_idx
            for sample in dataloaders[f"test{i}"]:
                inputs = sample[0]
                labels = sample[1]
                indexes = sample[2]

                inputs = inputs.to(device)
                feat1 = self.model1.logits(self.transform_test(inputs))
                preds_1 = feat1.argmax(dim=1)

                aug1 = self.transform_1(inputs)
                aug2 = self.transform_2(aug1)
                aug3 = self.transform_3(aug2)
                feat_aug = self.model1.logits(aug3)
                preds_aug = feat_aug.argmax(dim=1)
                probs_1 = torch.nn.functional.softmax(feat1, dim=1)

                # Grad-CAM Visualization
                target_layers = [self.model1.layer4[-1]]
                cam = GradCAM(model=self.model1, target_layers=target_layers)

                for j in range(inputs.size(0)):
                    if preds_1[j] == labels[j]:
                        input_img = inputs[j].unsqueeze(0) # [1, 3, 224, 224]
                        input_img_np = input_img.squeeze().cpu().numpy().transpose(1, 2, 0) # [224, 224, 3]
                        input_img_np = np.clip(input_img_np, 0, 1)

                        grayscale_cam = cam(input_tensor=input_img, targets=None) # ndarray,[1, 224, 224]
                        grayscale_cam = grayscale_cam[0, :]
                        cam_image = show_cam_on_image(input_img_np.astype(np.float32), grayscale_cam, use_rgb=True, image_weight=0.6)

                        aug3_img = aug3[j].unsqueeze(0)  # [1, 3, 224, 224]
                        aug3_img_np = aug3_img.squeeze().cpu().numpy().transpose(1, 2, 0)  # [224, 224, 3]
                        grayscale_cam_aug3 = cam(input_tensor=aug3_img, targets=None)  # ndarray,[1, 224, 224]
                        grayscale_cam_aug3 = grayscale_cam_aug3[0, :]
                        cam_image_aug3 = show_cam_on_image(aug3_img_np.astype(np.float32), grayscale_cam_aug3,
                                                           use_rgb=True, image_weight=0.6)

                        class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(labels[j])]
                        pred_class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(preds_1[j])]
                        pred_aug_class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(preds_aug[j])]
                        cam_image_pil = Image.fromarray(cam_image)
                        class_name = class_to_name[class_name]
                        if not os.path.exists('./cam/{}/RIGHT/'.format(self.params.learner)):
                            os.makedirs('./cam/{}/RIGHT/'.format(self.params.learner))
                        prob_1 = probs_1[j][preds_1[j]].item()
                        cam_image_pil.save('./cam/{}/RIGHT/{}_{}_{}.png'.format(self.params.learner, indexes[j], class_name, prob_1))
                    else:
                        input_img = inputs[j].unsqueeze(0)  # [1, 3, 224, 224]
                        input_img_np = input_img.squeeze().cpu().numpy().transpose(1, 2, 0)  # [224, 224, 3]
                        input_img_np = np.clip(input_img_np, 0, 1)

                        grayscale_cam = cam(input_tensor=input_img, targets=None)  # ndarray,[1, 224, 224]
                        grayscale_cam = grayscale_cam[0, :]
                        cam_image = show_cam_on_image(input_img_np.astype(np.float32), grayscale_cam, use_rgb=True,
                                                      image_weight=0.6)

                        aug3_img = aug3[j].unsqueeze(0)  # [1, 3, 224, 224]
                        aug3_img_np = aug3_img.squeeze().cpu().numpy().transpose(1, 2, 0)  # [224, 224, 3]
                        grayscale_cam_aug3 = cam(input_tensor=aug3_img, targets=None)  # ndarray,[1, 224, 224]
                        grayscale_cam_aug3 = grayscale_cam_aug3[0, :]
                        cam_image_aug3 = show_cam_on_image(aug3_img_np.astype(np.float32), grayscale_cam_aug3,
                                                           use_rgb=True, image_weight=0.6)

                        class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(labels[j])]
                        pred_class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(preds_1[j])]
                        pred_aug_class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(preds_aug[j])]
                        cam_image_pil = Image.fromarray(cam_image)
                        class_name = class_to_name[class_name]
                        pred_class_name = class_to_name[pred_class_name]
                        if not os.path.exists('./cam/{}/ERROR/'.format(self.params.learner)):
                            os.makedirs('./cam/{}/ERROR/'.format(self.params.learner))
                        cam_image_pil.save('./cam/{}/ERROR/{}_{}_{}.png'.format(self.params.learner, indexes[j], class_name, pred_class_name))
