import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from kamo.data import Standardizer
from kamo_experiments.evaluation import (deterministic_corruption, evaluate_model,
                                fit_std_temperature, ordered_window_metadata)
from kamo_experiments.analysis import _assert_aligned, paired_block_bootstrap, summarize_results


class ConstantForecaster(torch.nn.Module):
    def __init__(self, horizon=2, std=1.0):
        super().__init__()
        self.horizon, self.std = horizon, std

    def forward(self, x, mask=None, return_diagnostics=False, **kwargs):
        mean = x[:, -1:, :].expand(-1, self.horizon, -1)
        std = torch.full_like(mean, self.std)
        diagnostics = {"reliability": mask.float() * 0.7,
                       "router_weights": torch.ones(len(x), x.shape[2], 3) / 3} if return_diagnostics else None
        return SimpleNamespace(mean=mean, std=std, diagnostics=diagnostics)


class PhaseDataset(Dataset):
    def __init__(self):
        self.window_ids = np.arange(13) + 100
        self.window_starts = np.arange(13) * 5 + 20
        self.group_ids = np.full(13, -1)

    def __len__(self):
        return 13

    def __getitem__(self, index):
        return {"x": torch.zeros(4, 1), "y": torch.full((2, 1), float(self.window_starts[index])),
                "x_mask": torch.ones(4, 1).bool(), "y_mask": torch.ones(2, 1).bool(),
                "window_start": int(self.window_starts[index]), "window_id": int(self.window_ids[index]), "group_id": -1}


class PhaseForecaster(torch.nn.Module):
    def forward(self, x, cycle_index, **kwargs):
        mean = cycle_index.float()[:, None, None].expand(-1, 2, 1)
        return SimpleNamespace(mean=mean, std=torch.ones_like(mean), diagnostics=None)


class EvaluationTests(unittest.TestCase):
    def dataset(self):
        x = torch.arange(13 * 4 * 2).reshape(13, 4, 2).float() / 100
        y = x[:, -1:, :].expand(-1, 2, -1).clone() + 2
        xmask = torch.ones_like(x).bool()
        ymask = torch.ones_like(y).bool()
        ymask[0] = False  # Keep empty-window identity for paired alignment.
        ymask[2, 1, 1] = False
        y[2, 1, 1] = float("nan")  # Masked targets must never enter scores.
        return TensorDataset(x, y, xmask, ymask)

    def test_full_coverage_masks_global_rmse_and_prediction_shards(self):
        loader = DataLoader(self.dataset(), batch_size=2, shuffle=False)
        scaler = Standardizer(mean=np.array([10., 20.]), std=np.array([2., 3.]))
        with tempfile.TemporaryDirectory() as folder:
            metrics, arrays = evaluate_model(ConstantForecaster(), loader, "cpu", scaler, output_dir=folder, save_predictions=True)
            self.assertEqual(metrics["windows"], 13)
            self.assertEqual(metrics["observed_targets"], 47)
            self.assertEqual(metrics["zero_target_windows"], 1)
            self.assertEqual(len(arrays["crps"]), 13)
            self.assertTrue(np.isnan(arrays["crps"][0]))
            self.assertAlmostEqual(metrics["normalized_mse"], 4, places=5)
            self.assertAlmostEqual(metrics["rmse"], math.sqrt(metrics["mse"]), places=8)
            self.assertEqual(len(arrays["diag_reliability_mean"]), 13)
            shards = list((Path(folder) / "prediction_chunks").glob("*.npz"))
            self.assertEqual(len(shards), 7)  # Explicitly greater than old first-four limit.
            self.assertEqual(sum(len(np.load(p)["window_id"]) for p in shards), 13)
            self.assertTrue(json.loads((Path(folder) / "metrics.json").read_text())["complete"])

    def test_batch_size_and_training_rng_do_not_change_corruption(self):
        dataset = self.dataset()
        torch.manual_seed(1)
        ma, aa = evaluate_model(ConstantForecaster(), DataLoader(dataset, batch_size=2), "cpu", condition="impulse:0.20:5")
        torch.manual_seed(99999)
        mb, bb = evaluate_model(ConstantForecaster(), DataLoader(dataset, batch_size=5), "cpu", condition="impulse:0.20:5")
        np.testing.assert_allclose(aa["crps"], bb["crps"], equal_nan=True)
        np.testing.assert_array_equal(aa["target_hash"], bb["target_hash"])
        self.assertAlmostEqual(ma["crps"], mb["crps"], places=12)

    def test_zero_severity_and_gaussian_paired_draws(self):
        x = torch.ones(3, 7, 2)
        for condition in ("block_missing:0", "burst_impulse:0:5", "gaussian:0", "random_missing:0"):
            corrupted, mask = deterministic_corruption(x, condition, [5, 6, 7])
            torch.testing.assert_allclose(corrupted, x)
            self.assertTrue(mask.all())
        a, _ = deterministic_corruption(x, "gaussian:0.1", [5, 6, 7])
        b, _ = deterministic_corruption(x, "gaussian:0.2", [5, 6, 7])
        torch.testing.assert_allclose(b - x, 2 * (a - x), atol=2e-7, rtol=1e-5)

    def test_calibration_is_closed_form_and_masks_are_respected(self):
        result = fit_std_temperature(ConstantForecaster(), DataLoader(self.dataset(), batch_size=3), "cpu")
        self.assertAlmostEqual(result["std_temperature"], 2, places=6)
        self.assertEqual(result["observations"], 47)
        raw, _ = evaluate_model(ConstantForecaster(), DataLoader(self.dataset(), batch_size=3), "cpu")
        calibrated, _ = evaluate_model(ConstantForecaster(), DataLoader(self.dataset(), batch_size=3), "cpu", std_temperature=result["std_temperature"])
        self.assertLess(calibrated["nll"], raw["nll"])
        self.assertAlmostEqual(calibrated["mae"], raw["mae"])

    def test_dict_phase_is_original_timestamp(self):
        dataset = PhaseDataset()
        loader = DataLoader(dataset, batch_size=3)
        metrics, arrays = evaluate_model(PhaseForecaster(), loader, "cpu")
        self.assertEqual(metrics["mae"], 0)
        np.testing.assert_array_equal(arrays["window_id"], dataset.window_ids)
        np.testing.assert_array_equal(arrays["time_start"], dataset.window_starts)
        fit = fit_std_temperature(PhaseForecaster(), loader, "cpu")
        self.assertEqual(fit["std_temperature"], 0.05)

    def test_refuse_shuffled_or_truncated_evaluation(self):
        with self.assertRaises(ValueError):
            ordered_window_metadata(DataLoader(self.dataset(), batch_size=2, shuffle=True))
        with self.assertRaises(ValueError):
            ordered_window_metadata(DataLoader(self.dataset(), batch_size=2, drop_last=True))

    def test_invalid_observed_predictions_raise(self):
        with self.assertRaises(FloatingPointError):
            evaluate_model(ConstantForecaster(std=float("nan")), DataLoader(self.dataset(), batch_size=3), "cpu")

    def test_actual_kamo_and_cyclical_factory_full_evaluation(self):
        from kamo_experiments.factory import build_model
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            bundle = SimpleNamespace(num_vars=3, seq_len=16, pred_len=4, adjacency=None, dataset_name="ETTh1")
            config = dict(dataset="ETTh1", seq_len=16, pred_len=4, d_model=8, mixer_rank=2,
                          kamo_patch_scales="4,8", n_heads=2, n_layers=1, kamo_patch_layers=1,
                          ff_mult=2, patch_len=4, patch_stride=2, dropout=0., num_scales=2,
                          robust_window=3, huber_delta=3.)
            x, y = torch.randn(11, 16, 3), torch.randn(11, 4, 3)
            loader = DataLoader(TensorDataset(x, y), batch_size=2)
            for name in ("kamo_s", "cyclenet", "dlinear"):
                with self.subTest(model=name):
                    model = build_model(dict(config, model=name), bundle)
                    metrics, arrays = evaluate_model(model, loader, "cpu", condition="level_shift:1")
                    self.assertEqual(metrics["windows"], 11)
                    self.assertEqual(len(arrays["crps"]), 11)
                    self.assertTrue(np.isfinite(metrics["normalized_crps"]))
                    if name == "kamo_s":
                        self.assertIn("diag_reliability_mean", arrays)
                        self.assertIn("diag_router_weights_component_0", arrays)
        finally:
            torch.set_num_threads(previous)


class AnalysisTests(unittest.TestCase):
    def test_constant_paired_effect_exact_under_block_resampling(self):
        comparator = np.linspace(1, 2, 120)
        reference = comparator - 0.25
        counts = np.arange(120) % 7 + 1
        result = paired_block_bootstrap(reference, comparator, counts, block_length=17, n_boot=300)
        self.assertAlmostEqual(result["difference_ci_low"], -0.25, places=10)
        self.assertAlmostEqual(result["difference_ci_high"], -0.25, places=10)
        self.assertGreater(result["relative_gain_pct"], 0)

    def test_clusters_and_empty_windows(self):
        a = np.array([np.nan, 1, 2, 4, 3, 7.])
        b = a + 1
        counts = np.array([0, 2, 4, 3, 1, 7])
        result = paired_block_bootstrap(a, b, counts, n_boot=300, group_ids=[0, 0, 1, 1, 2, 2])
        self.assertEqual(result["method"], "patient_cluster_observation_weighted")
        self.assertAlmostEqual(result["difference_ci_low"], -1, places=10)
        self.assertEqual(result["resampling_units"], 3)

    def test_pairing_checks_targets_not_just_number_of_windows(self):
        a = dict(window_id=np.arange(3), time_start=np.arange(3), target_count=np.ones(3), target_hash=np.array([1, 2, 3]))
        b = dict(a, target_hash=np.array([3, 2, 1]))
        with self.assertRaises(ValueError):
            _assert_aligned(a, b)

    def test_report_integration_with_full_window_statistics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = PhaseDataset()
            for model, forecaster in (("kamo_s", PhaseForecaster()), ("baseline", ConstantForecaster())):
                for seed in (2026, 2027):
                    for condition in ("clean", "gaussian:0.10"):
                        output = root / model / str(seed) / condition.replace(":", "_")
                        metrics, _ = evaluate_model(forecaster, DataLoader(dataset, batch_size=3), "cpu", condition=condition, output_dir=output)
                        metrics.update(dataset="toy", model=model, seed=seed, seq_len=4, pred_len=2)
                        (output / "metrics.json").write_text(json.dumps(metrics))
            result = summarize_results(root, root / "report", bootstrap_replicates=100, block_length=3)
            self.assertEqual(result["audit"]["records"], 8)
            self.assertEqual(result["audit"]["paired_rows"], 6)
            self.assertTrue((root / "report" / "seed_summary.csv").exists())
            self.assertTrue((root / "report" / "clean_robustness_tradeoff.png").exists())


if __name__ == "__main__":
    unittest.main()
