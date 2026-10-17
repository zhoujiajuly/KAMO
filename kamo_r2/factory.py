from __future__ import annotations
from types import SimpleNamespace
import torch
from torch import nn
import legacy_train
from kamo.robustifier import RobustifiedInputWrapper

ALIASES = {
    "kamo_s": "kamo_v5_shared", "kamo_f": "kamo_v5",
    "kamo_separate": "kamo_v5_separate", "kamo_mask": "kamo_v5_mask_only",
    "kamo_no_scale": "kamo_v5_shared_no_unrel_scale",
    "kamo_no_norm": "kamo_v5_shared_no_rel_norm",
    "kamo_no_routing": "kamo_v5_shared_no_rel_routing",
    "kamo_no_gate": "kamo_v5_shared_no_rel_gate",
    "kamo_no_mixer": "kamo_v5_shared_no_mixer",
    "kamo_dense_mixer": "kamo_v5_shared_dense_mixer",
}
NEW_MODELS = {"dlinear", "nlinear", "cyclenet", "tqnet"}


class LegacyAdapter(nn.Module):
    def __init__(self, model):
        super().__init__(); self.backbone = model

    def forward(self, x, mask=None, cycle_index=None, **kwargs):
        return self.backbone(x, mask=mask, **kwargs)

    def optimizer_groups(self, learning_rate):
        if hasattr(self.backbone, "optimizer_groups"):
            return self.backbone.optimizer_groups(learning_rate)
        return self.backbone.parameters()


def make_args(config):
    args = legacy_train.parse_args([])
    vars(args).update(config)
    return args


def build_model(config, bundle):
    args = make_args(config)
    if args.model in NEW_MODELS:
        from .models import build_new_model
        return build_new_model(args.model, args, bundle)
    name = args.model
    args.model = ALIASES.get(name, name)
    if name == "timemixer_robust":
        args.model = "timemixer"
        return LegacyAdapter(RobustifiedInputWrapper(legacy_train.build_model(args, bundle),
                                                     args.robust_window, args.huber_delta))
    return LegacyAdapter(legacy_train.build_model(args, bundle))
