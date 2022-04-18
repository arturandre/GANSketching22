# -*- coding: utf-8 -*-
# @Date    : 10/23/21
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

from typing import Tuple, List
import logging

import torch
import torch.nn as nn

from modules import init_weights, freeze_params, GlobalAdaptiveAvgPool2d

logger = logging.getLogger(__name__)

HYPER_HEAD_REGISTRY = {}
HYPER_HEAD_CLASS_NAMES = set()


def register_hyper_head(name):
    """Registers a :class:`nn.Module` subclass.

    This decorator allows Classy Vision to instantiate a subclass of
    :class:`ClassyModel` from a configuration file, even if the class itself is
    not part of the Classy Vision framework. To use it, apply this decorator to
    a ClassyModel subclass, like this:

    .. code-block:: python

      @register_model('resnet')
      class ResidualNet(ClassyModel):
         ...

    To instantiate a model from a configuration file, see
    :func:`build_model`."""

    def register_model_cls(cls):
        if name in HYPER_HEAD_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))
        if cls.__name__ in HYPER_HEAD_CLASS_NAMES:
            raise ValueError(
                "Cannot register model with duplicate class name ({})".format(
                    cls.__name__
                )
            )
        HYPER_HEAD_REGISTRY[name] = cls
        HYPER_HEAD_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_model_cls

def build_hyper_head(name, **kwargs):
    """Builds a hypernet head from a config.

    This assumes a 'name' key in the config which is used to determine what
    model class to instantiate. For instance, a config `{"name": "my_model",
    "foo": "bar"}` will find a class that was registered as "my_model"
    (see :func:`register_model`) and call .from_config on it."""

    assert name.lower() in HYPER_HEAD_REGISTRY, "unknown model"
    model = HYPER_HEAD_REGISTRY[name.lower()].from_config(name, **kwargs)
    return model

class BaseHyperNetHead(nn.Module):
    def __init__(self, num_shot):
        super(BaseHyperNetHead, self).__init__()
        self.num_shot = num_shot
        self.weight_predictor: nn.Module
        self.bias_predictor: nn.Module

    def process_weight(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate the output affine weights.
        :param x: The sketch+augmentations of each x shall be [a, a^1, ..., a^29, b, b^1, ..., b^29, ...]
        :return: The sketch-wise style weights.
        """
        N, C, S, S = x.shape
        """
        N = Batch*30
        C = channels
        S = spatial dim of x
        """
        assert S == 1 # What this means?
        num_class = N // self.num_shot
        weights = []
        for sketch_idx in range(num_class):
            per_class_weight = x[
                sketch_idx * self.num_shot : (sketch_idx + 1) * self.num_shot
            ]
            per_class_weight = torch.mean(per_class_weight, dim=0, keepdim=True)
            weights.append(per_class_weight)
        return torch.cat(weights, dim=0)

    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the affine weight and bias.
        :param feature: shape is N * C * S * S. N = num_shot * num_class.
        :return: A Tuple. Each tensor's shape is num_class * C * S * S
        """
        assert feature.shape[0] % self.num_shot == 0
        weight = self.weight_predictor(feature)
        bias = self.bias_predictor(feature)

        return self.process_weight(weight), self.process_weight(bias)

@register_hyper_head("v1")
class HyperNetHeadv1(BaseHyperNetHead):
    """
    This block is intended to be attached to the discriminator.
    v1 use 3*3 and early output feature from backbone.
    """

    def __init__(self, name, num_shot, in_channel, out_channel):
        super().__init__(num_shot)
        self.name = name
        mid_channel = max(int(in_channel * 0.5), out_channel)
        #mid_channel = min(int(in_channel * 0.5), out_channel.numel())
        #out_channel = int(out_channel.numel()*0.1)
        self.weight_predictor = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(mid_channel, out_channel, kernel_size=1),
            GlobalAdaptiveAvgPool2d(),
        )
        self.bias_predictor = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(mid_channel, out_channel, kernel_size=1),
            GlobalAdaptiveAvgPool2d(),
        )

    @classmethod
    def from_config(cls, name, **kwargs) -> "HyperNetHeadv1":
        required_args = ["num_shot", "in_channel", "out_channel"]
        for arg in required_args:
            assert arg in kwargs.keys(), "HyperNetHeadv1 requires argument %s" % arg
        return cls(name, **kwargs)


class HyperNet(nn.Module):
    def __init__(
        self,
        num_support_shot: int,
        backbone: nn.Module,
        output_style_widths: List[int],
        freeze_backbone: bool,
        init_type: str = "normal",
        ):
        """
        This hyper network will predict several sketch-specific style weights at different level.
        :param num_support_shot: Number of shot in support set.
        :param backbone: Usually to be a discriminator.
        :param output_style_widths: The list of the predicted style weight width.
        """
        super().__init__()
        self.backbone = backbone
        backbone_out_ch = backbone.hyper_out_ch
        self.heads = nn.ModuleList(
            [
                build_hyper_head(
                    #cfg.HEAD,
                    "v1",
                    num_shot=num_support_shot,
                    in_channel=backbone_out_ch,
                    out_channel=style_width,
                )
                for style_width in output_style_widths
            ]
        )
        init_weights(self, init_type)
        self.freeze_backbone(freeze_backbone)

        pass

    def freeze_backbone(self, mode: bool) -> "HyperNet":
        if mode:
            print(f"=> freeze hyper network backbone")
            freeze_params(self.backbone)
        return self

    def forward(self, support_images: torch.Tensor) -> Tuple[Tuple]:
        feature = self.backbone.forward_backbone(support_images)
        # feature = self.backbone.inter_feats[-1]
        output = []
        for head in self.heads:
            output.append(head(feature))
        output = tuple(output)
        return output