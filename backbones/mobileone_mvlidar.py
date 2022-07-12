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


class IdentityConv2D(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(IdentityConv2D, self).__init__()
        self.weight = None
        self.bias = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

    def forward(self, x):
        if self.weight:
            x = nn.functional.conv2d(x, self.weight, self.bias,
                                     stride=self.stride,
                                     padding=self.padding,
                                     groups=self.groups)
        return x


class ConvDW(nn.Module):
    def __init__(self, in_channels, out_channels, stride, alpha_in=1.0, alpha_out=1.0, beta=1, eps=1e-5):
        super(ConvDW, self).__init__()
        self.fused_layers = False
        self.in_channels = int(alpha_in * in_channels)
        self.out_channels = int(alpha_out * out_channels)
        self.stride = stride
        self.eps = eps

        self.beta = beta
        self.dw_blocks = nn.ModuleList()

        for i in range(beta):
            self.dw_blocks.append(nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, 3, stride, 1, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels, eps=eps)))

        if stride == 1:
            self.dw_blocks.append(nn.Sequential(
                IdentityConv2D(),
                nn.BatchNorm2d(self.in_channels, eps=eps)))
            self.dw_blocks.append(nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, 1, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels, eps=eps)))

        self.fusion_blocks = nn.ModuleList()

        for i in range(beta):
            self.fusion_blocks.append(nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False),
                nn.BatchNorm2d(self.out_channels, eps=eps)))

        if self.in_channels == self.out_channels:
            self.fusion_blocks.append(nn.Sequential(
                IdentityConv2D(),
                nn.BatchNorm2d(self.out_channels, eps=eps)))

    def fusing_layers(self):
        device = torch.device('cpu')
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                device = module.weight.device
                break
        num = self.beta
        if self.stride == 1:
            self.dw_blocks[self.beta][0].weight = torch.zeros(size=(self.in_channels, 1, 3, 3), device=device)
            self.dw_blocks[self.beta][0].weight[:, :, 1, 1] = 1.0

            conv_1x1_weight = self.dw_blocks[self.beta + 1][0].weight.clone()
            self.dw_blocks[self.beta + 1][0].weight = torch.nn.Parameter(
                torch.zeros(size=(self.in_channels, 1, 3, 3), device=device))
            self.dw_blocks[self.beta + 1][0].weight.data[:, :, 1, 1] = conv_1x1_weight.view(-1, 1)
            num += 2

        for i in range(num):
            cur_conv_w = self.dw_blocks[i][0].weight
            cur_b_mean = self.dw_blocks[i][1].running_mean.view(-1, 1, 1, 1)
            cur_b_var = self.dw_blocks[i][1].running_var.view(-1, 1, 1, 1)
            cur_b_weight = self.dw_blocks[i][1].weight.view(-1, 1, 1, 1)
            cur_b_bias = self.dw_blocks[i][1].bias.view(-1, 1, 1, 1)

            w = cur_conv_w * cur_b_weight / (torch.sqrt(cur_b_var + self.eps))
            b = cur_b_bias - cur_b_mean * cur_b_weight / (torch.sqrt(cur_b_var + self.eps))

            if i == 0:
                weight = w
                bias = b
            else:
                weight += w
                bias += b

        module = nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1, groups=self.in_channels,
                           stride=self.stride, bias=True)
        module.weight.data = weight
        module.bias.data = bias.view(-1, )
        self.add_module('dw_blocks_fusion', module)

        num = self.beta
        if self.in_channels == self.out_channels:
            self.fusion_blocks[self.beta][0].weight = torch.zeros(size=(self.out_channels, self.out_channels, 1, 1),
                                                                  device=device)
            for channel_idx in range(self.in_channels):
                self.fusion_blocks[self.beta][0].weight.data[channel_idx, channel_idx] = 1.0
            num += 1

        for i in range(num):
            cur_conv_w = self.fusion_blocks[i][0].weight
            cur_b_mean = self.fusion_blocks[i][1].running_mean.view(-1, 1, 1, 1)
            cur_b_var = self.fusion_blocks[i][1].running_var.view(-1, 1, 1, 1)
            cur_b_weight = self.fusion_blocks[i][1].weight.view(-1, 1, 1, 1)
            cur_b_bias = self.fusion_blocks[i][1].bias.view(-1, 1, 1, 1)

            w = cur_conv_w * cur_b_weight / (torch.sqrt(cur_b_var + self.eps))
            b = cur_b_bias - cur_b_mean * cur_b_weight / (torch.sqrt(cur_b_var + self.eps))

            if i == 0:
                weight = w
                bias = b
            else:
                weight += w
                bias += b

        module = nn.Conv2d(self.out_channels, self.out_channels, 1, padding=0, groups=1, bias=True)
        module.weight.data = weight
        module.bias.data = bias.view(-1, )
        self.add_module('conv1x1_blocks_fusion', module)

    def forward_train(self, x):
        features = None
        for i in range(len(self.dw_blocks)):
            if i == 0:
                features = self.dw_blocks[i](x)
            else:
                features += self.dw_blocks[i](x)

        features = nn.functional.relu(features)

        fused = None
        for i, block in enumerate(self.fusion_blocks):
            if i == 0:
                fused = block(features)
            else:
                fused += block(features)

        fused = nn.functional.relu(fused)

        return fused

    def forward_export(self, x):

        x = self.dw_blocks_fusion(x)
        x = nn.functional.relu(x)

        x = self.conv1x1_blocks_fusion(x)
        x = nn.functional.relu(x)

        return x

    def forward(self, x):
        if self.fused_layers:
            return self.forward_export(x)
        else:
            return self.forward_train(x)


class MVLiDARNetSeg(torch.nn.Module):
    def __init__(self, ch_in, alpha=[1]*6, beta=4):
        super(MVLiDARNetSeg, self).__init__()

        self.block1 = ConvDW(ch_in, 64, 2, 1.0, alpha[0], beta)
        self.block2 = nn.Sequential(
            ConvDW(64, 64, 1, alpha[0], alpha[1], beta),
            ConvDW(64, 64, 2, alpha[1], alpha[1], beta))

        self.block3 = nn.Sequential(
            ConvDW(64, 128, 1, alpha[1], alpha[2], beta),
            ConvDW(128, 128, 1, alpha[2], alpha[2], beta),
            ConvDW(128, 128, 1, alpha[2], alpha[2], beta),
            ConvDW(128, 128, 1, alpha[2], alpha[2], beta),
            ConvDW(128, 128, 1, alpha[2], alpha[2], beta),
            ConvDW(128, 128, 1, alpha[2], alpha[2], beta),
            ConvDW(128, 128, 1, alpha[2], alpha[2], beta),
            ConvDW(128, 128, 2, alpha[2], alpha[2], beta))


        self.up1a = upconv(int(alpha[2] * 128), 256)
        self.up1c = conv1x1(256 + int(alpha[1] * 64), 256)
        self.up1d = conv3x3(256, 256)

        self.up2a = upconv(256, 128)
        self.up2c = conv1x1(128 + int(alpha[0] * 64), 128)
        self.up2d = conv3x3(128, 128)

        self.up3a = upconv(128, 64)
        self.up3b = conv1x1(64, 64)
        self.up3c = conv3x3(64, 64)

        self.classhead = nn.Sequential(conv3x3(64, 64),
                                       conv1x1(64, 20))

    def fusing_layers(self):
        for module in self.modules():
            if isinstance(module, ConvDW):
                module.fused_layers = True
                module.fusing_layers()

    def forward(self, feat):
        block1_feat = self.block1(feat)
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


def mobileone_s0(in_channels):
    alpha = [0.75, 0.75, 1.0, 1.0, 1.0, 2.0]
    model = MVLiDARNetSeg(ch_in=in_channels, alpha=alpha, beta=4)

    return model

def mobileone_s1(in_channels):
    alpha = [1.5, 1.5, 1.5, 2.0, 2.0, 2.5]
    model = MVLiDARNetSeg(ch_in=in_channels, alpha=alpha, beta=1)

    return model

def mobileone_s2(in_channels):
    alpha = [1.5, 1.5, 2.0, 2.5, 2.5, 4.0]
    model = MVLiDARNetSeg(ch_in=in_channels, alpha=alpha, beta=1)

    return model

def mobileone_s3(in_channels):
    alpha = [2.0, 2.0, 2.5, 3.0, 3.0, 4.0]
    model = MVLiDARNetSeg(ch_in=in_channels, alpha=alpha, beta=1)

    return model
def mobileone_s4(in_channels):
    alpha = [3.0, 3.0, 3.5, 3.5, 3.5, 4.0]
    model = MVLiDARNetSeg(ch_in=in_channels,alpha=alpha, beta=1)

    return model




# if __name__=='__main__':
#
#     inputs = torch.ones(size=(1, 3, 96, 960))
#     in_channel = 3
#     model = mobileone_s4(in_channel)
#
#     model.eval()
#     model.fusing_layers()
#     tensor_1 = model(inputs)
#     print("done")
#
#     torch.onnx.export(model, (inputs,), 'mobileone_s0.onnx', verbose=True, training=False,
#                       operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
#                       opset_version=10, input_names=['inputs'])

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






