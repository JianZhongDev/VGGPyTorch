"""
FILENAME: VGG.py
DESCRIPTION: Model definition for VGG
@author: Jian Zhong
"""

import torch
from torch import nn

from .Layers import StackedLayers
from .Layers import DebugLayers


# VGG model definition
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

        # add debug layers if debuging is enabled
        if enable_debug:
            stacked_layers += [
                DebugLayers.PrintMessageLayer("input"),
                DebugLayers.PrintShapeLayer()
            ]

        # add stacked convolutional layers and max pooling layers
        for i_stackconv_descrip in range(len(stacked_conv_descriptors)):
            cur_stacked_conv_descriptor = stacked_conv_descriptors[i_stackconv_descrip]
            if not isinstance(cur_stacked_conv_descriptor, list):
                continue
            stacked_layers.append(
                StackedLayers.VGGStacked2DConv(
                    cur_stacked_conv_descriptor
                )
            )

            # add debug layers if debuging is enabled
            if enable_debug:
                stacked_layers += [
                    DebugLayers.PrintMessageLayer("stacked conv"),
                    DebugLayers.PrintShapeLayer()
                ]

            # add max pooling layer after stacked convolutional layer
            stacked_layers.append(
                nn.MaxPool2d(
                    kernel_size = 2,
                    stride = 2,
                )
            )

            # add debug layers if debuging is enabled
            if enable_debug:
                stacked_layers += [
                    DebugLayers.PrintMessageLayer("max pool"),
                    DebugLayers.PrintShapeLayer()
                ]

        # flatten convolutional layers 
        stacked_layers.append(
            nn.Flatten()
        )

        # add debug layers if debuging is enabled
        if enable_debug:
            stacked_layers += [
                DebugLayers.PrintMessageLayer("flatten"),
                DebugLayers.PrintShapeLayer()
            ]
        
        # add stacked linear layers
        stacked_layers.append(
            StackedLayers.VGGStackedLinear(
                stacked_linear_descriptor
            )
        )

        # add debug layers if debuging is enabled
        if enable_debug:
            stacked_layers += [
                DebugLayers.PrintMessageLayer("stacked linear"),
                DebugLayers.PrintShapeLayer()
            ]

        # add softmax layer at the very end
        stacked_layers.append(
            nn.Softmax(dim = -1)
        )

        # add debug layers if debuging is enabled
        if enable_debug:
            stacked_layers += [
                DebugLayers.PrintMessageLayer("softmax"),
                DebugLayers.PrintShapeLayer(),
            ]

        # convert list of layers to Sequantial network
        if len(stacked_layers) > 0:
            self.network = nn.Sequential(*stacked_layers)

    def forward(self, x):
        y = self.network(x)
        return y


        

