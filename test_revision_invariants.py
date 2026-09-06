from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from calibrate_std_temperature import closed_form_temperature
from calibrate_std_multimetric import normalized_temperatures, select_temperature
from calibrate_validation import Candidate, candidate_std
from collect_revision_results import gate_decision
from kamo.baselines import PatchTSTForecaster
from kamo.engine import corrupt_context, gaussian_nll, masked_mae
from kamo.metrics import regression_metrics
from kamo.model import KAMOV5, VariableStateMixer
from kamo.robustifier import (
    ReliabilityConditionedWrapper,
    ReliabilitySoftInputWrapper,
    forward_fill,
)
from run_revision_suite import (
    TrainJob,
    evaluation_protocol,
    high_dimensional_jobs,
    long_horizon_jobs,
    mixer_jobs,
    real_missing_jobs,
    role_ablation_jobs,
    sensitivity_jobs,
    training_evidence_complete,
    transfer_jobs,
)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_kamo(mode: str, ablation: str = "none") -> KAMOV5:
    return KAMOV5(
        num_vars=5,
        seq_len=24,
        pred_len=12,
        d_model=16,
        mixer_rank=4,
        patch_scales=(4, 8),
        n_heads=2,
        reliability_mode=mode,
        reliability_hidden=8,
        dropout=0.0,
        ablation=ablation,
    )


def test_vectorized_causal_state_matches_reference_loops() -> None:
    torch.manual_seed(5)
    values = torch.randn(3, 24, 5)
    observed = torch.rand(3, 24, 5) > 0.3
    observed[:, :3, 0] = False
    observed[:, 7:12, 2] = False

    reference_fill = torch.zeros_like(values)
    last = torch.zeros_like(values[:, 0])
    for step in range(values.shape[1]):
        last = torch.where(observed[:, step], values[:, step], last)
        reference_fill[:, step] = last

    reference_age = torch.zeros_like(observed, dtype=torch.float32)
    running = torch.zeros_like(reference_age[:, 0])
    for step in range(observed.shape[1]):
        running = torch.where(
            observed[:, step], torch.zeros_like(running), running + 1.0
        )
        reference_age[:, step] = running
    reference_age /= observed.shape[1]

    assert torch.equal(KAMOV5._forward_fill(values, observed), reference_fill)
    assert torch.equal(forward_fill(values, observed), reference_fill)
    assert torch.equal(KAMOV5._missing_age(observed), reference_age)


def test_shared_and_separate_are_parameter_matched() -> None:
    shared = make_kamo("shared_learned")
    separate = make_kamo("separate_learned")
    mask_only = make_kamo("mask_only")
    assert parameter_count(shared) == parameter_count(separate)
    assert parameter_count(shared) == parameter_count(mask_only)


def test_shared_constraint_is_real() -> None:
    torch.manual_seed(7)
    shared = make_kamo("shared_learned").eval()
    separate = make_kamo("separate_learned").eval()
    separate.load_state_dict(shared.state_dict())
    with torch.no_grad():
        shared.reliability_controller.network[-1].bias.copy_(
            torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        )
        separate.reliability_controller.network[-1].bias.copy_(
            torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        )
    x = torch.randn(2, 24, 5)
    x[:, 8, 2] = 15.0
    shared_roles = shared(x, return_diagnostics=True).diagnostics["role_reliability"]
    separate_roles = separate(x, return_diagnostics=True).diagnostics["role_reliability"]
    assert torch.allclose(shared_roles[..., 0], shared_roles[..., 4])
    assert not torch.allclose(separate_roles[..., 0], separate_roles[..., 4])


def test_routing_ablation_removes_only_reliability_input() -> None:
    torch.manual_seed(19)
    full = make_kamo("fixed_shared").eval()
    no_rel_routing = make_kamo(
        "fixed_shared", ablation="no_reliability_routing"
    ).eval()
    no_rel_routing.load_state_dict(full.state_dict())
    assert parameter_count(full) == parameter_count(no_rel_routing)
    assert parameter_count(no_rel_routing.router) > 0

    x = 0.1 * torch.randn(2, 24, 5)
    x[:, 12, 2] = 25.0
    observed = torch.ones_like(x, dtype=torch.bool)
    with torch.no_grad():
        full_diag = full(x, mask=observed, return_diagnostics=True).diagnostics
        ablated_diag = no_rel_routing(
            x, mask=observed, return_diagnostics=True
        ).diagnostics

    assert torch.any(ablated_diag["reliability"] < 0.99)
    assert torch.allclose(
        full_diag["routing_reliability"], full_diag["reliability"]
    )
    assert torch.equal(
        ablated_diag["routing_reliability"], observed.to(x.dtype)
    )


def test_soft_reliability_strength_has_interpretable_endpoints() -> None:
    torch.manual_seed(29)
    full = make_kamo("fixed_shared").eval()
    softened = make_kamo("fixed_shared").eval()
    softened.load_state_dict(full.state_dict())
    softened.reliability_strength = 0.0
    x = 0.1 * torch.randn(2, 24, 5)
    x[:, 12, 2] = 25.0
    observed = torch.ones_like(x, dtype=torch.bool)
    with torch.no_grad():
        full_diag = full(x, mask=observed, return_diagnostics=True).diagnostics
        soft_diag = softened(x, mask=observed, return_diagnostics=True).diagnostics
    assert torch.any(full_diag["routing_reliability"] < 0.99)
    assert torch.equal(soft_diag["routing_reliability"], observed.to(x.dtype))


def test_uncertainty_reliability_strength_zero_disables_only_inflation() -> None:
    model = make_kamo("fixed_shared").eval()
    model.uncertainty_reliability_strength = 0.0
    x = 0.1 * torch.randn(2, 24, 5)
    x[:, 12, 2] = 25.0
    with torch.no_grad():
        diagnostics = model(x, return_diagnostics=True).diagnostics
    assert torch.any(diagnostics["corruption_fraction"] > 0)
    assert torch.count_nonzero(diagnostics["scale_fraction"]) == 0


def test_calibration_std_formula_is_positive_and_monotone() -> None:
    base = np.ones((2, 3, 1), dtype=np.float64)
    signal = np.zeros_like(base)
    fraction = np.full((2, 1, 1), 0.5, dtype=np.float64)
    low = Candidate(1.0, 0.0, 0.0)
    high = Candidate(1.0, 0.0, 1.0)
    low_std = candidate_std(base, signal, fraction, 0.5, low, temperature=1.2)
    high_std = candidate_std(base, signal, fraction, 0.5, high, temperature=1.2)
    assert np.all(low_std > 0)
    assert np.all(high_std > low_std)


def test_closed_form_temperature_matches_gaussian_nll_optimum() -> None:
    raw, selected = closed_form_temperature(20.0, 2, 0.05, 100.0)
    assert math.isclose(raw, math.sqrt(10.0))
    assert math.isclose(selected, raw)


def test_closed_form_temperature_respects_safety_bounds() -> None:
    _, low = closed_form_temperature(0.0, 2, 0.05, 100.0)
    _, high = closed_form_temperature(80000.0, 2, 0.05, 100.0)
    assert low == 0.05
    assert high == 100.0


def test_multimetric_temperature_requires_identity_candidate() -> None:
    assert normalized_temperatures((2.0, 1.0, 1.0)) == (1.0, 2.0)
    try:
        normalized_temperatures((0.5, 2.0))
    except ValueError:
        pass
    else:
        raise AssertionError("A calibration grid without T=1 must be rejected")


def test_multimetric_temperature_rejects_nll_only_overcorrection() -> None:
    scores = (
        {
            "temperature": 1.0,
            "nll": 10.0,
            "crps": 2.0,
            "interval_score_95": 5.0,
            "calibration_error_95": 0.05,
        },
        {
            "temperature": 1.1,
            "nll": 8.0,
            "crps": 2.01,
            "interval_score_95": 5.02,
            "calibration_error_95": 0.04,
        },
        {
            "temperature": 4.0,
            "nll": 3.0,
            "crps": 3.0,
            "interval_score_95": 8.0,
            "calibration_error_95": 0.01,
        },
    )
    selected, feasible = select_temperature(scores, 0.01, 0.01)
    assert selected["temperature"] == 1.1
    assert {row["temperature"] for row in feasible} == {1.0, 1.1}


def test_postgate_kamo_jobs_use_shared_learned_main_variant() -> None:
    job_groups = (
        role_ablation_jobs(),
        long_horizon_jobs(),
        high_dimensional_jobs(False),
        real_missing_jobs(),
        mixer_jobs(),
        sensitivity_jobs(),
    )
    for jobs in job_groups:
        kamo_jobs = [job for job in jobs if job.model.startswith("kamo_v5")]
        assert kamo_jobs
        assert all(job.model.startswith("kamo_v5_shared") for job in kamo_jobs)


def test_robust_statistics_are_causal_within_context() -> None:
    model = make_kamo("fixed_shared").eval()
    prefix = torch.randn(2, 24, 5)
    altered = prefix.clone()
    altered[:, 18:, :] += 100.0
    with torch.no_grad():
        first = model(prefix, return_diagnostics=True).diagnostics
        second = model(altered, return_diagnostics=True).diagnostics
    # A change beginning at context index 18 must not alter robust statistics
    # at earlier timestamps.  This guards against centered/non-causal windows.
    for key in ("local_median", "robust_scale", "reliability"):
        assert torch.allclose(first[key][:, :18], second[key][:, :18])


def test_no_mixer_is_identity() -> None:
    mixer = VariableStateMixer(5, rank=0, adjacency=None, mixer_type="none")
    state = torch.randn(3, 5, 8)
    assert torch.equal(mixer(state), state)


def test_masked_losses_ignore_unobserved_targets() -> None:
    y = torch.tensor([[[1.0], [1000.0]]])
    mean = torch.zeros_like(y)
    std = torch.ones_like(y)
    mask = torch.tensor([[[True], [False]]])
    assert torch.allclose(masked_mae(y, mean, mask), torch.tensor(1.0))
    assert gaussian_nll(y, mean, std, mask) < 1.0


def test_masked_probabilistic_metrics_and_tail_interval() -> None:
    y = np.array([[[0.0], [999.0]]])
    mean = np.zeros_like(y)
    std = np.ones_like(y)
    mask = np.array([[[True], [False]]])
    metrics = regression_metrics(y, mean, std, mask)
    assert metrics["mae"] == 0.0
    assert metrics["picp_99"] == 1.0
    assert "interval_score_95" in metrics


def test_unseen_corruptions_are_seeded() -> None:
    x = torch.randn(4, 24, 5)
    for condition in (
        "sensor_freeze:0.5:0.25",
        "gradual_drift:2.0",
        "heavy_tail:0.2:10.0",
    ):
        first, first_mask = corrupt_context(x, condition, seed=123)
        second, second_mask = corrupt_context(x, condition, seed=123)
        assert torch.equal(first, second)
        assert torch.equal(first_mask, second_mask)


def test_portable_reliability_wrapper_shapes() -> None:
    backbone = PatchTSTForecaster(
        num_vars=5,
        seq_len=24,
        pred_len=12,
        d_model=16,
        patch_len=8,
        stride=4,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
    )
    model = ReliabilityConditionedWrapper(backbone).eval()
    output = model(torch.randn(2, 24, 5), return_diagnostics=True)
    assert output.mean.shape == (2, 12, 5)
    assert output.std.shape == (2, 12, 5)
    assert torch.all(output.std > 0)
    assert "reliability" in output.diagnostics


def test_soft_reliability_adapter_is_parameter_matched_and_bounded() -> None:
    backbone = PatchTSTForecaster(
        num_vars=3,
        seq_len=24,
        pred_len=12,
        d_model=16,
        patch_len=8,
        stride=4,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
    )
    backbone_parameters = parameter_count(backbone)
    model = ReliabilitySoftInputWrapper(backbone).eval()
    assert parameter_count(model) == backbone_parameters

    x = torch.randn(2, 24, 3)
    x[:, 12, 1] += 50.0
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, 5:8, 2] = False
    output = model(x, mask=mask, return_diagnostics=True)
    diagnostics = output.diagnostics
    soft = diagnostics["soft_values"]
    filled = diagnostics["filled_values"]
    robust = diagnostics["robust_values"]
    lower = torch.minimum(filled, robust) - 1e-6
    upper = torch.maximum(filled, robust) + 1e-6
    assert torch.all((soft >= lower) & (soft <= upper))
    assert output.mean.shape == (2, 12, 3)
    assert output.std.shape == (2, 12, 3)
    assert torch.all(torch.isfinite(output.mean))
    assert torch.all(output.std > 0)


def test_formal_transfer_jobs_use_soft_adapter_not_legacy_adapter() -> None:
    models = {job.model for job in transfer_jobs()}
    assert "patchtst_reliability_soft" in models
    assert "itransformer_reliability_soft" in models
    assert "patchtst_reliability" not in models
    assert "itransformer_reliability" not in models


def test_formal_transfer_evaluation_uses_frozen_multimetric_calibration() -> None:
    transfer = TrainJob("transfer", "weather", "patchtst_reliability_soft", 2026)
    other = TrainJob("highdim", "electricity", "patchtst", 2026)
    assert evaluation_protocol(transfer) == (
        "transfer_multimetric",
        "std-multimetric-auto",
    )
    assert evaluation_protocol(other) == ("highdim", "none")


def test_validation_only_checkpoint_is_reusable_without_test_access() -> None:
    job = TrainJob("transfer", "weather", "patchtst_reliability_soft", 2026)
    valid = {
        "dataset": "weather",
        "model": "patchtst_reliability_soft",
        "seed": 2026,
        "selection_split": "validation",
        "selection_metric": "crps",
        "best_val_crps": 14.123,
        "best_epoch": 32,
        "test_accessed": False,
    }
    assert training_evidence_complete(job, selection=valid)
    assert not training_evidence_complete(job, selection={**valid, "test_accessed": True})
    assert not training_evidence_complete(job, selection={**valid, "seed": 2027})


def test_gate_decision_requires_complete_parameter_matched_evidence() -> None:
    rows = []
    values = {
        "kamo_v5_shared": {"clean": 1.0, "corrupt": 0.90},
        "kamo_v5_separate": {"clean": 1.0, "corrupt": 0.905},
        "kamo_v5_mask_only": {"clean": 1.0, "corrupt": 1.00},
    }
    for dataset in ("ETTh2", "weather"):
        for model, scores in values.items():
            for seed in (2026, 2027, 2028):
                for condition in ("clean", "impulse:0.05:5.0"):
                    rows.append(
                        {
                            "family": "gate",
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "condition": condition,
                            "crps": scores["clean" if condition == "clean" else "corrupt"],
                            "full_test": True,
                        }
                    )
    decision = gate_decision(pd.DataFrame(rows))
    assert decision["complete"] is True
    assert decision["pass"] is True


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
