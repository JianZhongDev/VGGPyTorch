"""
FILENAME: Evaluate.py
DESCRIPTION: VGG performance evoluation functions
@author: Jian Zhong
"""

import torch

# Evaluate if label is within top k prediction result
def batch_in_top_k(outputs, labels, top_k = 1):
    sorted_outputs, sorted_idxs = torch.sort(outputs, dim = -1, descending = True)
    in_top_k = torch.full_like(labels, False)
    for cur_idx in range(top_k):
        in_top_k = torch.logical_or(sorted_idxs[:,cur_idx] == labels, in_top_k)
    return in_top_k    

