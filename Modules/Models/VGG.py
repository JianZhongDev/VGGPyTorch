"""
FILENAME: VGG.py
DESCRIPTION: Model definition for VGG
@author: Jian Zhong
"""

import torch
from torch import nn

from .Layers import StackedLayers
from .Layers import DebugLayers


class VGG(nn.Module):
    def __init__(
        self,
        stacked_conv_descriptors,
        stacked_linear_descriptor,
        enable_debug = False,
    ):
        assert(isinstance(stacked_conv_descriptors, list))
        assert(isinstance(stacked_linear_descriptor, list))
        super().__init__()

        self.network = nn.Identity()

        stacked_layers = []

        if enable_debug:
            stacked_layers += [
                DebugLayers.PrintMessageLayer("input"),
                DebugLayers.PrintShapeLayer()
            ]
        for i_stackconv_descrip in range(len(stacked_conv_descriptors)):
            cur_stacked_conv_descriptor = stacked_conv_descriptors[i_stackconv_descrip]
            if not isinstance(cur_stacked_conv_descriptor, list):
                continue
            stacked_layers.append(
                StackedLayers.VGGStacked2DConv(
                    cur_stacked_conv_descriptor
                )
            )
            if enable_debug:
                stacked_layers += [
                    DebugLayers.PrintMessageLayer("stacked conv"),
                    DebugLayers.PrintShapeLayer()
                ]
            stacked_layers.append(
                nn.MaxPool2d(
                    kernel_size = 2,
                    stride = 2,
                )
            )
            if enable_debug:
                stacked_layers += [
                    DebugLayers.PrintMessageLayer("max pool"),
                    DebugLayers.PrintShapeLayer()
                ]
        stacked_layers.append(
            nn.Flatten()
        )
        if enable_debug:
            stacked_layers += [
                DebugLayers.PrintMessageLayer("flatten"),
                DebugLayers.PrintShapeLayer()
            ]
        stacked_layers.append(
            StackedLayers.VGGStackedLinear(
                stacked_linear_descriptor
            )
        )
        if enable_debug:
            stacked_layers += [
                DebugLayers.PrintMessageLayer("stacked linear"),
                DebugLayers.PrintShapeLayer()
            ]
        stacked_layers.append(
            nn.Softmax(dim = -1)
        )
        if enable_debug:
            stacked_layers += [
                DebugLayers.PrintMessageLayer("softmax"),
                DebugLayers.PrintShapeLayer(),
            ]

        if len(stacked_layers) > 0:
            self.network = nn.Sequential(*stacked_layers)

    def forward(self, x):
        y = self.network(x)
        return y


        

