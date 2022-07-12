#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tasks.semantic.postproc.CRF import CRF
import __init__ as booger

# pytorc_lightning

class Segmentator(nn.Module):
  def __init__(self, ARCH, nclasses, path=None, path_append="", strict=False):
    super().__init__()
    self.ARCH = ARCH
    self.nclasses = nclasses
    self.path = path
    self.path_append = path_append
    self.strict = False

    # get the model
    bboneModule = imp.load_source("bboneModule",
                                  booger.TRAIN_PATH + '/backbones/' +
                                  self.ARCH["backbone"]["name"] + '.py')
    # self.backbone = bboneModule.mobileone_s0(in_channels=3)
    # self.backbone = bboneModule.mobileone_s1(in_channels=3)
    # self.backbone = bboneModule.mobileone_s2(in_channels=3)
    # self.backbone = bboneModule.mobileone_s3(in_channels=3)
    self.backbone = bboneModule.mobileone_s4(in_channels=3)


    if torch.cuda.is_available():
      self.backbone.cuda()

    # train backbone?
    if not self.ARCH["backbone"]["train"]:
      for w in self.backbone.parameters():
        w.requires_grad = False


    # print number of parameters and the ones requiring gradients
    # print number of parameters and the ones requiring gradients
    weights_total = sum(p.numel() for p in self.parameters())
    weights_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print("Total number of parameters: ", weights_total)
    print("Total number of parameters requires_grad: ", weights_grad)

    # breakdown by layer
    weights_enc = sum(p.numel() for p in self.backbone.parameters())
    print("Param encoder ", weights_enc)

    # get weights
    if path is not None:
      # try backbone
      try:
        w_dict = torch.load(path + "/backbone" + path_append,
                            map_location=lambda storage, loc: storage)
        self.backbone.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model backbone weights")
      except Exception as e:
        print()
        print("Couldn't load backbone, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e


  def forward(self, x, mask=None):
    y  = self.backbone(x)
    y = F.softmax(y, dim=1)

    return y

  def save_checkpoint(self, logdir, suffix=""):
    # Save the weights
    torch.save(self.backbone.state_dict(), logdir +
               "/backbone" + suffix)

