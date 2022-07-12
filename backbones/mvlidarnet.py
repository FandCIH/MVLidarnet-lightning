# -*- coding: utf-8 -*-
"""
Created on Mon May 30 17:33:12 2022

@author: novauto
"""

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, stride=stride, bias=False),
              nn.BatchNorm2d(out_channels),
              nn.ReLU()]
    return nn.Sequential(*layers)

def conv1x1(in_channels, out_channels):
    
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False),
              nn.BatchNorm2d(out_channels),
              nn.ReLU()]
    return nn.Sequential(*layers)

def upconv(in_channels, out_channels):
    
    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU()]
    return nn.Sequential(*layers)



class Inception(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1):
        super(Inception, self).__init__()
        
        mid_channels = in_channels // 2

        self.branch_0 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)

        self.branch_1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                                      nn.ReLU(),
                                     nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
                                     )

        self.branch_2 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                                      nn.ReLU(),
                                     nn.Conv2d(mid_channels, mid_channels * 3, kernel_size=3, padding=1, bias=False),
                                     nn.ReLU(),
                                     nn.Conv2d(mid_channels * 3, mid_channels * 3, kernel_size=3, padding=1, bias=False),
                                     )
        self.branch_3 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False))
        
        self.conv = conv3x3(mid_channels * 6, out_channels, stride=stride, padding=1)

    def forward(self, x):

        branch_0 = self.branch_0(x)

        branch_1 = self.branch_1(x)

        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)
        
        x = torch.cat([branch_0, branch_1, branch_2, branch_3], 1)
        x = self.conv(x)

        return x
    
class MVLiDARNetSeg(torch.nn.Module):
    def __init__(self):
        super(MVLiDARNetSeg, self).__init__()
        
        self.trunk = nn.Sequential(conv3x3(3, 64),
                                 conv3x3(64, 64),
                                 conv3x3(64, 128, 2))
        
        
        # self.block1 =nn.Sequential(Inception(128, 64),
        #         #                            Inception(64, 64, 2))

        self.block1 = nn.Sequential(Inception(128, 64),
                                    Inception(64, 64))
        
        self.block2 = nn.Sequential(Inception(64, 64),
                                   Inception(64, 64, 2))
        
        self.block3 = nn.Sequential(Inception(64, 64),
                                   Inception(64, 64),
                                   Inception(64, 128, 2))

        
        self.up1a = upconv(128, 256)
        self.up1c = conv1x1(320, 256)
        self.up1d = conv3x3(256, 256)
        
        self.up2a = upconv(256, 128)
        self.up2c = conv1x1(192, 128)
        self.up2d = conv3x3(128, 128)
        
        self.up3a = upconv(128, 64)
        self.up3b = conv1x1(64, 64)
        self.up3c = conv3x3(64, 64)

        self.classhead = nn.Sequential(conv3x3(64, 64),
                                 conv1x1(64, 7))

    def forward(self, feat):
        
        
        trunk_feat = self.trunk(feat)
        
        block1_feat = self.block1(trunk_feat)
        block2_feat = self.block2(block1_feat)
        block3_feat = self.block3(block2_feat)

        f_up1a = self.up1a(block3_feat)
        f_up1b = torch.cat([f_up1a, block2_feat], 1)
        f_up1c = self.up1c(f_up1b)
        f_up1d = self.up1d(f_up1c)
        
        f_up2a = self.up2a(f_up1d)
        f_up2b = torch.cat([f_up2a, block1_feat], 1)
        f_up2c = self.up2c(f_up2b)
        f_up2d = self.up2d(f_up2c)
        
        f_up3a = self.up3a(f_up2d)
        f_up3b = self.up3b(f_up3a)
        f_up3c = self.up3c(f_up3b)
        
        classhead = self.classhead(f_up3c)
        return classhead
    

# model_seg = MVLiDARNetSeg()

# seg_feat = torch.ones(size=(2, 3, 64, 2048)).float()
# model_seg(seg_feat)
# model_seg.eval()
#
# # 分析FLOPs
# flops = FlopCountAnalysis(model_seg, (seg_feat))
# print("FLOPs: ", flops.total()/1e9)
#
# seg_feat = torch.ones(size=(1, 3, 96, 960)).float()
# model_seg(seg_feat)
# model_seg.eval()
#
# # 分析FLOPs
# flops = FlopCountAnalysis(model_seg, (seg_feat))
# print("FLOPs: ", flops.total()/1e9)

#
# torch.onnx.export(model_seg, seg_feat, 'mvlidarseg.onnx', verbose=True, training=False,
#                               operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=9,
#                               input_names=['seg_feat'])





        
        