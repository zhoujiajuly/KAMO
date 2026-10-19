"""Split leakage, data identity and model-control tests using synthetic data only."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from kamo_experiments.data import build_bundle, EpisodeWindows
from kamo_experiments.factory import ALIASES, build_model


def _write_csv(root, values):
    frame = pd.DataFrame(values, columns=["feature_%d" % i for i in range(values.shape[1])])
    frame.insert(0, "date", pd.date_range("2020-01-01", periods=len(frame), freq="h"))
    frame.to_csv(Path(root) / "ETTh1.csv", index=False)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = self.directory.name
        time = np.arange(400, dtype=np.float32)[:, None]
        self.values = time * np.arange(1, 9, dtype=np.float32)[None] + np.arange(8)[None]
        _write_csv(self.root, self.values)

    def tearDown(self):
        self.directory.cleanup()

    def bundle(self, **kwargs):
        return build_bundle("ETTh1", self.root, seq_len=16, pred_len=4,
                            batch_size=7, workers=0, **kwargs)

    def test_every_target_timestamp_is_disjoint_across_splits(self):
        bundle = self.bundle()
        datasets = [loader.dataset for loader in (bundle.train, bundle.val, bundle.calibration, bundle.test)]
        # Histories legitimately overlap earlier partitions. It is all target
        # timestamps (including horizon endpoints), not just starts, that must
        # be disjoint across train, validation selection, calibration and test.
        expected_bounds = [(16, 280), (280, 300), (300, 320), (320, 400)]
        target_sets = []
        for ds, (lower, upper) in zip(datasets, expected_bounds):
            target_times = set()
            for start in ds.window_starts:
                first = int(start) + ds.seq_len
                last = first + ds.pred_len - 1
                self.assertGreaterEqual(first, lower)
                self.assertLess(last, upper)
                target_times.update(range(first, last + 1))
            self.assertEqual(min(target_times), lower)
            self.assertEqual(max(target_times), upper - 1)
            target_sets.append(target_times)
        for i in range(len(target_sets)):
            for j in range(i + 1, len(target_sets)):
                self.assertFalse(target_sets[i] & target_sets[j])
        self.assertLess(int(datasets[1].window_starts[0]), 280)

    def test_scaler_fits_only_observed_training_values(self):
        values = self.values.copy()
        values[5, 0] = np.nan
        values[19, 1] = np.inf
        values[281:, :] = 1e7
        _write_csv(self.root, values)
        bundle = self.bundle()
        finite_train = np.where(np.isfinite(values[:280]), values[:280], np.nan)
        np.testing.assert_allclose(bundle.scaler.mean[0], np.nanmean(finite_train.astype(np.float64), axis=0), rtol=1e-6)
        np.testing.assert_allclose(bundle.scaler.std[0], np.nanstd(finite_train.astype(np.float64), axis=0), rtol=1e-6)
        original_mean = bundle.scaler.mean.copy()
        original_std = bundle.scaler.std.copy()
        values[280:] = -1e8
        _write_csv(self.root, values)
        changed = self.bundle()
        np.testing.assert_array_equal(changed.scaler.mean, original_mean)
        np.testing.assert_array_equal(changed.scaler.std, original_std)

    def test_missing_masks_preserved_and_missing_targets_excluded(self):
        values = self.values.copy()
        values[5, 0] = np.nan
        values[18, 2] = np.inf
        _write_csv(self.root, values)
        bundle = self.bundle()
        example = bundle.train.dataset[0]
        self.assertFalse(bool(example["x_mask"][5, 0]))
        self.assertFalse(bool(example["y_mask"][2, 2]))
        self.assertEqual(float(example["x"][5, 0]), 0.0)
        self.assertEqual(float(example["y"][2, 2]), 0.0)
        self.assertTrue(bool(torch.isfinite(example["x"]).all()))
        self.assertTrue(bool(torch.isfinite(example["y"]).all()))
        self.assertEqual(bundle.audit["missing_values"], 2)
        # A future value cannot fill a missing observation: the mask and zero
        # remain even if the next observed value is arbitrarily changed.
        values[6, 0] = 1e6
        _write_csv(self.root, values)
        after = self.bundle().train.dataset[0]
        self.assertFalse(bool(after["x_mask"][5, 0]))
        self.assertEqual(float(after["x"][5, 0]), 0.0)

    def test_dimensional_subsets_are_nested_reproducible_and_realign_columns(self):
        small = self.bundle(dimensions=3, feature_seed=77)
        large = self.bundle(dimensions=6, feature_seed=77)
        repeated = self.bundle(dimensions=3, feature_seed=77)
        s_cols, l_cols = small.audit["columns"], large.audit["columns"]
        self.assertTrue(set(s_cols).issubset(set(l_cols)))
        self.assertEqual(s_cols, sorted(s_cols))
        self.assertEqual(s_cols, repeated.audit["columns"])
        np.testing.assert_array_equal(small.train.dataset.values, self.values[:, s_cols])
        self.assertEqual(small.audit["data_sha256"], repeated.audit["data_sha256"])
        self.assertNotEqual(small.audit["data_sha256"], large.audit["data_sha256"])

    def test_content_fingerprint_is_path_and_loader_independent_but_detects_changes(self):
        first = self.bundle()
        again = build_bundle("etth1", self.root, seq_len=16, pred_len=4, batch_size=13)
        self.assertEqual(first.audit["data_sha256"], again.audit["data_sha256"])
        with tempfile.TemporaryDirectory() as other_root:
            _write_csv(other_root, self.values)
            moved = build_bundle("ETTh1", other_root, seq_len=16, pred_len=4, batch_size=7)
            self.assertEqual(first.audit["data_sha256"], moved.audit["data_sha256"])
        changed = self.values.copy()
        changed[-1, -1] += 10
        _write_csv(self.root, changed)
        self.assertNotEqual(first.audit["data_sha256"], self.bundle().audit["data_sha256"])

    def test_phase_is_actual_absolute_window_start_not_batch_index(self):
        bundle = self.bundle()
        batch = next(iter(bundle.test))
        self.assertEqual(batch["window_start"].tolist(), list(range(304, 311)))
        self.assertEqual(batch["window_id"].tolist(), list(range(304, 311)))
        for row in range(batch["x"].shape[0]):
            original = bundle.scaler.inverse(batch["x"][row].numpy())
            start = int(batch["window_start"][row])
            np.testing.assert_allclose(original, self.values[start:start + 16], atol=1e-3)

    def test_feature_with_no_training_observations_rejected(self):
        values = self.values.copy()
        values[:280, 2] = np.nan
        _write_csv(self.root, values)
        with self.assertRaisesRegex(ValueError, "no finite training"):
            self.bundle()


class FactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self, name):
        return dict(model=name, dataset="ETTh1", seq_len=16, pred_len=4,
                    d_model=8, mixer_rank=2, kamo_patch_scales="4,8",
                    n_heads=2, n_layers=1, kamo_patch_layers=1,
                    ff_mult=2, patch_len=4, patch_stride=2, dropout=0.0,
                    num_scales=2, robust_window=3, huber_delta=3.0)

    def bundle(self):
        return SimpleNamespace(num_vars=3, seq_len=16, pred_len=4,
                               adjacency=None, dataset_name="ETTh1")

    def test_all_control_aliases_forward_and_backward(self):
        expected = {
            "kamo_s": ("shared_learned", "none", "low_rank"),
            "kamo_f": ("fixed_shared", "none", "low_rank"),
            "kamo_separate": ("separate_learned", "none", "low_rank"),
            "kamo_mask": ("mask_only", "none", "low_rank"),
            "kamo_no_scale": ("shared_learned", "no_unreliability_scale", "low_rank"),
            "kamo_no_norm": ("shared_learned", "no_reliability_normalization", "low_rank"),
            "kamo_no_routing": ("shared_learned", "no_reliability_routing", "low_rank"),
            "kamo_no_gate": ("shared_learned", "no_reliability_residual_gate", "low_rank"),
            "kamo_no_mixer": ("shared_learned", "none", "none"),
            "kamo_dense_mixer": ("shared_learned", "none", "dense"),
        }
        self.assertEqual(set(expected), set(ALIASES))
        for name in ALIASES:
            with self.subTest(model=name):
                model = build_model(self.config(name), self.bundle())
                backbone = model.backbone
                self.assertEqual((backbone.reliability_mode, backbone.ablation, backbone.state_mixer.mixer_type), expected[name])
                x = torch.randn(2, 16, 3)
                mask = torch.ones_like(x, dtype=torch.bool)
                mask[:, 5:7, 0] = False
                output = model(x, mask=mask, cycle_index=torch.tensor([20, 21]), return_diagnostics=True)
                self.assertEqual(tuple(output.mean.shape), (2, 4, 3))
                self.assertTrue(bool(torch.isfinite(output.mean).all()))
                self.assertTrue(bool((output.std > 0).all()))
                (output.mean.square().mean() + output.std.mean()).backward()
                grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in grads))

    def test_shared_separate_mask_controls_are_parameter_matched(self):
        models = [build_model(self.config(name), self.bundle())
                  for name in ("kamo_s", "kamo_separate", "kamo_mask")]
        counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
        self.assertEqual(len(set(counts)), 1)
        shapes = [{name: tuple(value.shape) for name, value in model.state_dict().items()} for model in models]
        self.assertEqual(shapes[0], shapes[1])
        self.assertEqual(shapes[0], shapes[2])

    def test_multiscale_robust_baseline_forward_and_backward(self):
        model = build_model(self.config("timemixer_robust"), self.bundle())
        x = torch.randn(2, 16, 3)
        x[:, 6, 1] = 50.0
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, 9:11, 2] = False
        before = x.clone()
        output = model(x, mask=mask, cycle_index=torch.tensor([29, 30]), return_diagnostics=True)
        self.assertEqual(tuple(output.mean.shape), (2, 4, 3))
        self.assertTrue(bool(torch.isfinite(output.mean).all()))
        self.assertTrue(bool((output.std > 0).all()))
        self.assertTrue(torch.equal(x, before))
        self.assertIn("robust_values", output.diagnostics)
        (output.mean.square().mean() + output.std.mean()).backward()
        grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(any(bool((grad.abs() > 0).any()) for grad in grads))
        self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in grads))


if __name__ == "__main__":
    unittest.main()
