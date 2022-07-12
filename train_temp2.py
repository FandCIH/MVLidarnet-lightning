aimport os
from argparse import ArgumentParser
import argparse
import subprocess
import datetime
import torch.nn as nn
import torch.optim as optim
import yaml
from shutil import copyfile
import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import imp
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import random
from modules.ioueval import *
from dataset.kitti.parser import *

# Sweep parameters
hyperparameter_defaults = dict(
    ARCH_YAML = '/home/liuzihao/lightning/mvlidarnet/config/arch/darknet21.yaml',
    data_dir = '/home/liuzihao/dataset/kitti_segmentation',
    DATA_YAML = 'config/labels/semantic-kitti.yaml',
    pretrained = None,
    n_classes = 20,
    log = 'log',
    path = None,
    path_append = "",
    batch_size=4,
    lr=1e-3,
    num_layers=5,
    features_start=64,
    bilinear=False,
    grad_batches=1,
    epochs=5
)

wandb.init(config=hyperparameter_defaults)
# Config parameters are automatically set by W&B sweep agent
config = wandb.config


class SegModel(pl.LightningModule):
    '''
    Semantic Segmentation Module
    This is a basic semantic segmentation module implemented with Lightning.
    It uses CrossEntropyLoss as the default loss function. May be replaced with
    other loss functions as required.
    It uses the FCN ResNet50 model as an example.
    Adam optimizer is used along with Cosine Annealing learning rate scheduler.
    '''

    def __init__(self, hparams, strict=False):
        super().__init__()
        ARCH = yaml.safe_load(open(hparams.ARCH_YAML, 'r'))
        DATA = yaml.safe_load(open(hparams.DATA_YAML, 'r'))
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = hparams.data_dir
        self.nclasses = hparams.n_classes
        self.path = hparams.path
        self.path_append = hparams.path_append
        self.strict = False
        self.lr = hparams.lr

        current_directory = os.path.dirname(os.path.abspath(__file__))
        parserPath = os.path.join(current_directory, "dataset", "kitti", "parser.py")
        parserModule = imp.load_source("parserModule", parserPath)
        self.parser = parserModule.Parser(root=self.datadir,
                                          train_sequences=self.DATA["split"]["train"],
                                          valid_sequences=self.DATA["split"]["valid"],
                                          test_sequences=None,
                                          labels=self.DATA["labels"],
                                          color_map=self.DATA["color_map"],
                                          learning_map=self.DATA["learning_map"],
                                          learning_map_inv=self.DATA["learning_map_inv"],
                                          sensor=self.ARCH["dataset"]["sensor"],
                                          max_points=self.ARCH["dataset"]["max_points"],
                                          batch_size=hparams.batch_size,
                                          workers=self.ARCH["train"]["workers"],
                                          gt=True,
                                          shuffle_train=True)
        # weights for loss (and bias)
        # weights for loss (and bias)
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.nclasses, dtype=torch.float)
        for cl, freq in self.DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)  # map actual class to xentropy class
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)  # get weights
        for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
            if self.DATA["learning_ignore"][x_cl]:
                # don't weigh
                self.loss_w[x_cl] = 0
        print(self.loss_w)

        # get the model
        bboneModule = imp.load_source("bboneModule",
                                      current_directory + "/backbones/" + self.ARCH["backbone"]["name"] + '.py')
        self.backbone = bboneModule.MVLiDARNetSeg()

        # if torch.cuda.is_available():
        #     self.backbone.cuda()

        # train backbone?
        if not self.ARCH["backbone"]["train"]:
            for w in self.backbone.parameters():
                w.requires_grad = False

        # print number of parameters and the ones requiring gradients
        # print number of parameters and the ones requiring gradients
        # weights_total = sum(p.numel() for p in self.parameters())
        # weights_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print("Total number of parameters: ", weights_total)
        # print("Total number of parameters requires_grad: ", weights_grad)
        #
        # # breakdown by layer
        # weights_enc = sum(p.numel() for p in self.backbone.parameters())
        # print("Param encoder ", weights_enc)

    def forward(self, x, mask=None):
        y = self.backbone(x)
        y = F.softmax(y, dim=1)

        return y



    def training_step(self, batch, batch_nb):

        #准备数据
        in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _ = batch

        output = self.backbone(in_vol)

        # 计算loss
        if "loss" in self.ARCH["train"].keys() and self.ARCH["train"]["loss"] == "xentropy":
            criterion = nn.NLLLoss(weight=self.loss_w)
        loss_val = criterion(torch.log(output.clamp(min=1e-8)), proj_labels)

        # 计算iou
        # self.ignore_class = []
        # for i, w in enumerate(self.loss_w):
        #     if w < 1e-10:
        #         self.ignore_class.append(i)
        #         print("Ignoring class ", i, " in IoU evaluation")
        # evaluator = iouEval(self.nclasses, self.device, self.ignore_class)

        # with torch.no_grad():
        #     evaluator.reset()
        #     argmax = output.argmax(dim=1)
        #     evaluator.addBatch(argmax, proj_labels)
        #     accuracy = evaluator.getacc()
        #     jaccard, class_jaccard = evaluator.getIoU()
        # acc = accuracy.update(accuracy.item(), in_vol.size(0))
        # iou = jaccard.update(jaccard.item(), in_vol.size(0))
        # self.log('train_loss', loss_val,
        #          'train_acc', acc,
        #          'train_iou', iou)  # log training loss

        self.log('train_loss', loss_val)
        return loss_val

    def validation_step(self, batch, batch_idx):
        # 准备数据
        in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _ = batch
        output = self(in_vol)

        # 计算loss
        if "loss" in self.ARCH["train"].keys() and self.ARCH["train"]["loss"] == "xentropy":
            criterion = nn.NLLLoss(weight=self.loss_w)
        loss_val = criterion(torch.log(output.clamp(min=1e-8)), proj_labels)

        # 计算iou
        # self.ignore_class = []
        # for i, w in enumerate(self.loss_w):
        #     if w < 1e-10:
        #         self.ignore_class.append(i)
        #         print("Ignoring class ", i, " in IoU evaluation")
        # evaluator = iouEval(self.nclasses, self.device, self.ignore_class)
        #
        # with torch.no_grad():
        #     evaluator.reset()
        #     argmax = output.argmax(dim=1)
        #     evaluator.addBatch(argmax, proj_labels)
        #     accuracy = evaluator.getacc()
        #     jaccard, class_jaccard = evaluator.getIoU()
        # acc = accuracy.update(accuracy.item(), in_vol.size(0))
        # iou = jaccard.update(jaccard.item(), in_vol.size(0))
        # self.log('val_loss', loss_val,
        #          'val_acc', acc,
        #          'val_iou', iou)  # log training loss

        self.log('train_loss', loss_val)
        return loss_val

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.backbone.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]


class KittiDataModule(pl.LightningModule):
    '''
    Kitti Data Module
    It is specific to KITTI dataset i.e. dataloaders are for KITTI
    and Normalize transform uses the mean and standard deviation of this dataset.
    '''

    def __init__(self, hparams):
        super().__init__()
        # open arch config file
        try:
            print("Opening arch config file %s" % hparams.ARCH_YAML)
            ARCH = yaml.safe_load(open(hparams.ARCH_YAML, 'r'))
        except Exception as e:
            print(e)
            print("Error opening arch yaml file.")
            quit()

        # open data config file
        try:
            print("Opening data config file %s" % hparams.DATA_YAML)
            DATA = yaml.safe_load(open(hparams.DATA_YAML, 'r'))
        except Exception as e:
            print(e)
            print("Error opening data yaml file.")
            quit()

        # create log folder
        try:
            if os.path.isdir(hparams.log):
                shutil.rmtree(hparams.log)
            os.makedirs(hparams.log)
        except Exception as e:
            print(e)
            print("Error creating log directory. Check permissions!")
            quit()

        # does model folder exist?
        if hparams.pretrained is not None:
            if os.path.isdir(hparams.pretrained):
                print("model folder exists! Using model from %s" % (hparams.pretrained))
            else:
                print("model folder doesnt exist! Start with random weights...")
        else:
            print("No pretrained directory found.")

        # copy all files to log folder (to remember what we did, and make inference
        # easier). Also, standardize name to be able to open it later
        try:
            print("Copying files to %s for further reference." % hparams.log)
            copyfile(hparams.ARCH_YAML, hparams.log + "/arch_cfg.yaml")
            copyfile(hparams.DATA_YAML, hparams.log + "/data_cfg.yaml")
        except Exception as e:
            print(e)
            print("Error copying files, check permissions. Exiting...")
            quit()
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = hparams.data_dir
        self.log = hparams.log
        self.path = hparams.pretrained
        self.shuffle_train = True
        self.batch_size = hparams.batch_size


        current_directory = os.path.dirname(os.path.abspath(__file__))
        parserPath = os.path.join(current_directory, "dataset", "kitti", "parser.py")
        parserModule = imp.load_source("parserModule", parserPath)
        self.parser = parserModule.Parser(root=self.datadir,
                                          train_sequences=self.DATA["split"]["train"],
                                          valid_sequences=self.DATA["split"]["valid"],
                                          test_sequences=None,
                                          labels=self.DATA["labels"],
                                          color_map=self.DATA["color_map"],
                                          learning_map=self.DATA["learning_map"],
                                          learning_map_inv=self.DATA["learning_map_inv"],
                                          sensor=self.ARCH["dataset"]["sensor"],
                                          max_points=self.ARCH["dataset"]["max_points"],
                                          batch_size=hparams.batch_size,
                                          workers=self.ARCH["train"]["workers"],
                                          gt=True,
                                          shuffle_train=True)

    def setup(self, stage=None):
        self.train_dataset = SemanticKitti(root=self.datadir,
                                           sequences=self.DATA["split"]["train"],
                                           labels=self.DATA["labels"],
                                           color_map=self.DATA["color_map"],
                                           learning_map=self.DATA["learning_map"],
                                           learning_map_inv=self.DATA["learning_map_inv"],
                                           sensor=self.ARCH["dataset"]["sensor"],
                                           max_points=self.ARCH["dataset"]["max_points"],
                                           gt=True)
        self.valid_dataset = SemanticKitti(root=self.datadir,
                                           sequences=self.DATA["split"]["valid"],
                                           labels=self.DATA["labels"],
                                           color_map=self.DATA["color_map"],
                                           learning_map=self.DATA["learning_map"],
                                           learning_map_inv=self.DATA["learning_map_inv"],
                                           sensor=self.ARCH["dataset"]["sensor"],
                                           max_points=self.ARCH["dataset"]["max_points"],
                                           gt=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=20,
                            pin_memory=True, drop_last=True)
    def val_dataloader(self):
        # return DataLoader(self.validset, batch_size=self.batch_size, shuffle=False, num_workers=20)
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=20,
                            pin_memory=True, drop_last=True)

def main(config):

    # ------------------------
    # 1 LIGHTNING MODEL
    # ------------------------
    model = SegModel(config)
    print(model)

    # ------------------------
    # 2 DATA PIPELINES
    # ------------------------
    kittiData = KittiDataModule(config)

    # for batch_id, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in enumerate(kittiData.train_dataloader()):
    #     print(in_vol.shape)

    #
    # in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _ = a


    # ------------------------
    # 3 WANDB LOGGER
    # ------------------------
    wandb.finish()
    wandb_logger = WandbLogger()
    #
    # # optional: log model topology
    wandb_logger.watch(model.backbone)

    # ------------------------
    # 4 TRAINER
    # ------------------------
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=config.epochs,
        accumulate_grad_batches=config.grad_batches,
        accelerator='gpu',
        strategy='dp',
        devices='2'
    )
    # #
    # # # ------------------------
    # # # 5 START TRAINING
    # # # ------------------------
    trainer.fit(model, kittiData)


if __name__ == '__main__':
    # FLAGS = parse_args()
    print(f'Starting a run with {config}')
    main(config)

