import yaml
from shutil import copyfile
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import imp
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from common.avgmeter import *
from modules.ioueval import *
from dataset.kitti.parser import *
import os

# Sweep parameters 数据均在42服务器上，文件夹 /home/liuzihao
hyperparameter_defaults = dict(
    ARCH_YAML='/home/liuzihao/lightning/mvlidarnet/config/arch/darknet21.yaml',
    data_dir='/home/liuzihao/dataset/kitti_segmentation',
    DATA_YAML='config/labels/semantic-kitti-7class.yaml',
    pretrained=None,
    n_classes=7,
    log='log',
    path=None,
    path_append="",
    num_workers=20,
    batch_size=4,
    lr=1e-3,
    grad_batches=1,
    max_epochs=50
)

# log文件
wandb.init(config=hyperparameter_defaults)
# Config parameters are automatically set by W&B sweep agent
config = wandb.config


# 模型
class SegModel(pl.LightningModule):
    '''
    Semantic Segmentation Module
    This is a basic semantic segmentation module implemented with Lightning.
    It uses CrossEntropyLoss as the default loss function. May be replaced with
    other loss functions as required.
    It uses the FCN ResNet50 model as an example.
    Adam optimizer is used along with Cosine Annealing learning rate scheduler.
    '''

    def __init__(self, hparams, parser, ARCH, DATA, strict=False):
        super().__init__()
        self.ARCH = ARCH
        self.DATA = DATA
        self.path = hparams.path
        self.path_append = hparams.path_append
        self.strict = False
        self.lr = hparams.lr
        self.parser = parser
        # 重新映射的类别数目
        self.nclasses = len(self.DATA["learning_map_inv"])
        current_directory = os.path.dirname(os.path.abspath(__file__))

        # weights for loss (and bias) 给每个类别加上权重
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = np.zeros(self.nclasses).astype(np.float32)
        for cl, freq in self.DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl) # map actual class to xentropy class
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)  # get weights
        for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
            if self.DATA["learning_ignore"][x_cl]:
                # don't weigh
                self.loss_w[x_cl] = 0.0
        print("Loss weights from content: ", self.loss_w)

        # get the model 这里可以更换backbone
        bboneModule = imp.load_source("bboneModule", current_directory + "/backbones/" + self.ARCH["backbone"]["name"] + '.py')
        self.backbone = bboneModule.MVLiDARNetSeg()

    def forward(self, x, mask=None):
        y = self.backbone(x)

        return y

    def training_step(self, batch, batch_nb):
        #准备数据
        in_vol,  proj_labels = batch
        output = self.backbone(in_vol)

        # 准备acc、iou
        acc = AverageMeter()
        iou = AverageMeter()
        loss = F.cross_entropy(output, proj_labels.long(), weight=torch.from_numpy(self.loss_w).cuda())

        # 计算iou
        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
        evaluator = iouEval(self.nclasses, self.device, self.ignore_class)
        with torch.no_grad():
            evaluator.reset()
            argmax = output.argmax(dim=1)
            evaluator.addBatch(argmax, proj_labels)
            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()
        acc.update(accuracy.item(), in_vol.size(0))
        iou.update(jaccard.item(), in_vol.size(0))

        values = {"train_loss": loss, "train_acc": acc.avg, "train_mIou:": iou.avg}
        self.log_dict(values)
        return loss

    def validation_step(self, batch, batch_idx):
        #准备数据
        in_vol,  proj_labels = batch
        output = self.backbone(in_vol)

        # 准备acc、iou
        acc = AverageMeter()
        iou = AverageMeter()
        loss = F.cross_entropy(output, proj_labels.long(), weight=torch.from_numpy(self.loss_w).cuda())

        # 计算iou
        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
        evaluator = iouEval(self.nclasses, self.device, self.ignore_class)
        with torch.no_grad():
            evaluator.reset()
            argmax = output.argmax(dim=1)
            evaluator.addBatch(argmax, proj_labels)
            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()
        acc.update(accuracy.item(), in_vol.size(0))
        iou.update(jaccard.item(), in_vol.size(0))

        # val后，展示每个类别的信息
        for i, jacc in enumerate(class_jaccard):
            print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=self.parser.get_xentropy_class_string(i), jacc=jacc))

        values = {"val_loss": loss, "val_acc": acc.avg, "val_mIou:": iou.avg}
        self.log_dict(values)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.backbone.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

# 创建DataLoader
class KittiDataModule(pl.LightningModule):
    '''
    Kitti Data Module
    It is specific to KITTI dataset i.e. dataloaders are for KITTI
    and Normalize transform uses the mean and standard deviation of this dataset.
    '''

    def __init__(self, hparams, parser, ARCH, DATA):
        super().__init__()
        self.ARCH = ARCH
        self.DATA = DATA
        self.data_dir = hparams.data_dir
        self.log = hparams.log
        # 预训练模型地址，暂时没有
        self.path = hparams.pretrained
        self.batch_size = hparams.batch_size
        self.num_workers = hparams.num_workers


    def train_dataloader(self):
        self.trainset = SemanticKitti(root=self.data_dir,
                                      sequences=self.DATA["split"]["train"],
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      gt=True)
        return torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                                           pin_memory=True, drop_last=True)
    def val_dataloader(self):
        self.validset = SemanticKitti(root=self.data_dir,
                                      sequences=self.DATA["split"]["valid"],
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      gt=True)
        return torch.utils.data.DataLoader(self.validset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                                        pin_memory=True, drop_last=True)

# 训练
class MyTrainer(pl.LightningModule):
    # 初始化函数，主要是读取模型的yaml和数据的yaml文件
    def __init__(self, hparams):
        super().__init__()
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

            # create log folder,
            # 这个log文件夹用于存放 yaml 文件和模型结果；wandb也是存放模型结果的，不过是框架自动存放，这个log可以存一些自己想存的数据，
            # 等wandb用熟悉了，可以把结果全放在wandb中
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
        self.data_dir = hparams.data_dir
        self.log = hparams.log
        # 预训练模型地址，暂时没有
        self.path = hparams.pretrained
        self.batch_size = hparams.batch_size
        self.num_workers = hparams.num_workers

        current_directory = os.path.dirname(os.path.abspath(__file__))
        parserPath = os.path.join(current_directory, "dataset", "kitti", "parser.py")
        parserModule = imp.load_source("parserModule", parserPath)
        self.parser = parserModule.Parser(root=self.data_dir,
                                     train_sequences=self.DATA["split"]["train"],
                                     valid_sequences=self.DATA["split"]["valid"],
                                     test_sequences=None,
                                     labels=self.DATA["labels"],
                                     color_map=self.DATA["color_map"],
                                     learning_map=self.DATA["learning_map"],
                                     learning_map_inv=self.DATA["learning_map_inv"],
                                     sensor=self.ARCH["dataset"]["sensor"],
                                     max_points=self.ARCH["dataset"]["max_points"],
                                     batch_size=self.batch_size,
                                     workers=self.num_workers,
                                     gt=True,
                                     shuffle_train=True)

    def train(self):
        # ------------------------
        # 1 LIGHTNING MODEL
        # ------------------------
        model = SegModel(config, self.parser, self.ARCH, self.DATA)
        # ------------------------
        # 2 DATA PIPELINES
        # ------------------------
        kittiData = KittiDataModule(config, self.parser, self.ARCH, self.DATA)
        # ------------------------
        # 3 WANDB LOGGER
        # ------------------------
        wandb.finish()
        wandb_logger = WandbLogger()
        #
        # # optional: log model topology
        wandb_logger.watch(model.backbone )
        # ------------------------
        # 4 TRAINER
        # ------------------------
        trainer = pl.Trainer(
            logger=wandb_logger,
            max_epochs=config.max_epochs,
            accumulate_grad_batches=config.grad_batches,
            accelerator='gpu',
            strategy='dp',
            devices='4'
        )
        # # # ------------------------
        # # # 5 START TRAINING
        # # # ------------------------
        trainer.fit(model, kittiData.train_dataloader(), kittiData.val_dataloader())

if __name__ == '__main__':
    print(f'Starting a run with {config}')
    trainer = MyTrainer(config)
    trainer.train()

