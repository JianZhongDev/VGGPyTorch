"""
FILENAME: DebugLayers.py
DESCRIPTION: customized layers used for debug
@author: Jian Zhong
"""

import torch
from torch import nn


class PrintShapeLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(f"DEBUG:\t x.size = {x.size()}")
        return x


class PrintValueLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(f"DEBUG:\t x = {x}")
        return x


class PrintMessageLayer(nn.Module):
    def __init__(self,message = ""):
        super().__init__()
        self.message = message
    
    def forward(self, x):
        print("DEBUG:\t MSG: " + self.message)
        return x