"""Predeclared recipes: no test-guided model or hyperparameter selection."""
from __future__ import annotations
import itertools
import random
import zlib
from .factory import NEW_MODELS

PROTOCOL_VERSION = "R2-20261008-v1"
CORE_DATASETS = ["ETTh1", "ETTh2", "weather", "exchange"]
CORE_MODELS = ["kamo_s", "dlinear", "cyclenet", "tqnet", "patchtst", "itransformer",
               "timemixer", "vpnet", "patchtst_robust", "itransformer_robust", "timemixer_robust"]
CONTROL_MODELS = ["kamo_f", "kamo_separate", "kamo_mask", "kamo_no_scale",
                  "kamo_no_norm", "kamo_no_routing", "kamo_no_gate"]
TUNE_CONDITIONS = ["clean", "gaussian:0.10", "impulse:0.01:5", "random_missing:0.10", "block_missing:0.10"]
SEEDS = [2026, 2027, 2028]


def search_space(model, dataset):
    if model in NEW_MODELS:
        from .models import new_model_search_space
        space = new_model_search_space(model, dataset)
        if model in {"dlinear", "nlinear"}:
            space["linear_individual"] = [False]  # genuinely lightweight shared linear model
        return space
    common = dict(lr=[1e-4, 3e-4, 1e-3, 3e-3], weight_decay=[0.0, 1e-4, 1e-3])
    if model.startswith("kamo"):
        common.update(d_model=[32, 48, 64], dropout=[0.0, 0.1, 0.2],
                      mixer_rank=[8, 16, 32], huber_delta=[2., 3., 4.])
    elif model.startswith("patchtst"):
        common.update(d_model=[64, 128, 256], n_layers=[1, 2, 3],
                      dropout=[0.0, 0.1, 0.2], patch_len=[8, 16, 32])
    elif model.startswith("itransformer"):
        common.update(d_model=[64, 128, 256], n_layers=[1, 2, 3], dropout=[0.0, 0.1, 0.2])
    elif model.startswith("timemixer"):
        common.update(d_model=[32, 64, 128], n_layers=[1, 2, 3],
                      num_scales=[2, 3], dropout=[0.0, 0.1, 0.2])
    elif model == "vpnet":
        common.update(d_model=[32, 64, 128], n_layers=[1, 2], patch_len=[8, 16],
                      vp_middle_size=[128, 256, 512], dropout=[0.0, 0.1, 0.2])
    else:
        raise ValueError("No frozen search space for " + model)
    if model.endswith("_robust"):
        common["huber_delta"] = [2., 3., 4.]
    return common


def reference_config(model, dataset):
    cfg = dict(model=model, dataset=dataset, lr=1e-3, weight_decay=1e-4, d_model=48,
               n_layers=2, dropout=.1, patch_len=16, patch_stride=8, mixer_rank=16,
               num_scales=3, huber_delta=3., linear_individual=False)
    if model in {"cyclenet", "tqnet"}:
        from .models import DEFAULT_CYCLES
        cfg.update(d_model=128, cycle_len=DEFAULT_CYCLES[dataset], cycle_model_type="mlp", use_revin=True)
    if model.startswith(("patchtst", "itransformer")):
        cfg["d_model"] = 128
    if model.startswith("timemixer"):
        cfg["d_model"] = 64
    return cfg


def candidates(model, dataset, count=12, num_vars=None):
    space = search_space(model, dataset)
    if model.startswith("kamo"):
        actual_d = num_vars or {"ETTh1": 7, "ETTh2": 7, "exchange": 8, "weather": 21,
                                "electricity": 321, "physionet2012": 37}.get(dataset, 321)
        space["mixer_rank"] = sorted(set(min(r, actual_d) for r in [2, 4, 8, 16]))
    keys = sorted(space)
    combinations = [dict(zip(keys, vals)) for vals in itertools.product(*(space[k] for k in keys))]
    rng = random.Random(73021 + zlib.crc32((model + "/" + dataset).encode()))
    rng.shuffle(combinations)
    ref = reference_config(model, dataset)
    if model.startswith("kamo"):
        ref["mixer_rank"] = min(ref["mixer_rank"], actual_d)
    projected = {k: ref.get(k, space[k][0]) for k in keys}
    combinations = [projected] + [c for c in combinations if c != projected]
    if count > len(combinations) or count < 1:
        raise ValueError("Requested {} trials but {} has {} distinct effective configs".format(count, model, len(combinations)))
    return [dict(ref, **c) for c in combinations[:count]]


def recipe_cases(recipe):
    def make(ds, model, l=96, h=96, d=None, inherited=False):
        return dict(dataset=ds, model=model, seq_len=l, pred_len=h, dimensions=d, inherited=inherited)
    if recipe == "core":
        return [make(d, m) for d in CORE_DATASETS for m in CORE_MODELS]
    if recipe == "controls":
        return [make(d, m, inherited=True) for d in CORE_DATASETS for m in CONTROL_MODELS]
    if recipe == "highdim":
        return [make("electricity", m) for m in ["kamo_s", "dlinear", "cyclenet", "tqnet", "patchtst_robust", "timemixer_robust"]]
    if recipe == "physionet":
        return [make("physionet2012", m, 24, 12) for m in ["kamo_s", "dlinear", "patchtst_robust", "itransformer_robust", "timemixer_robust"]]
    if recipe == "long":
        long_main = [make(d, m, 96, h) for d in ["ETTh2", "weather"] for h in [336, 720]
                     for m in ["kamo_s", "dlinear", "cyclenet", "tqnet", "patchtst_robust",
                               "timemixer_robust", "itransformer", "vpnet"]]
        h192_main = [make(d, m, 96, 192) for d in ["ETTh2", "weather"]
                     for m in ["kamo_s", "itransformer", "vpnet"]]
        return long_main + h192_main
    if recipe == "dimension":
        return [make("electricity", m, d=d) for d in [32, 64, 128]
                for m in ["kamo_s", "dlinear", "timemixer_robust"]]
    if recipe == "pilot":
        return [make("ETTh2", m) for m in ["kamo_s", "dlinear", "cyclenet", "tqnet", "timemixer_robust"]]
    raise ValueError("Unknown recipe " + recipe)


def case_name(case):
    return "{dataset}/{model}/L{seq_len}_H{pred_len}_D{dim}".format(dim=case.get("dimensions") or "all", **case)


def case_shard(case, shards):
    return zlib.crc32(case_name(case).encode()) % shards
