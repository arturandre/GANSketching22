# -*- coding: utf-8 -*-
# @Date    : 2/16/21
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

import collections
import logging
from itertools import repeat
from typing import List, Optional, Tuple, Type, Union  # noqa

import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def init_weights(model: nn.Module, init_type: str):
    init_type = init_type.lower()

    def _init_conv(m, init_type):
        if hasattr(m, "weight") and m.weight is not None:
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == "orth":
                nn.init.orthogonal_(m.weight.data)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform(m.weight.data, 1.0)
            else:
                raise NotImplementedError("{} unknown inital type".format(init_type))

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            _init_conv(m, init_type)
        elif isinstance(m, nn.BatchNorm2d):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)


def freeze_params(model: nn.Module):
    for (name, p) in model.named_parameters():
        p.requires_grad = False
        logger.info(f"freeze {name}")


class GlobalAdaptiveAvgPool2d(nn.Module):
    """
    Be used to synthesize either kernel weight (only 1X1) or bias.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        N = x.data.size(0)
        C = x.data.size(1)
        H, W = x.data.size(2), x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C, 1, 1)
        return x
