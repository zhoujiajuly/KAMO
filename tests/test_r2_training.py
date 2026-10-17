import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from kamo.data import Standardizer
from kamo_r2.data import SeriesWindows, make_loader
from kamo_r2.training import train_model
from kamo_r2 import training
from kamo_r2.storage import load_checkpoint


class TestTraining(unittest.TestCase):
    def bundle(self):
        values = np.random.RandomState(42).normal(size=(100, 3)).astype(np.float32)
        mask = np.ones_like(values, dtype=bool)
        scale = Standardizer(np.zeros((1, 3), dtype=np.float32), np.ones((1, 3), dtype=np.float32))
        train = SeriesWindows(values, mask, np.arange(0, 12), 16, 8, scale)
        val = SeriesWindows(values, mask, np.arange(40, 45), 16, 8, scale)
        return SimpleNamespace(num_vars=3, seq_len=16, pred_len=8, adjacency=None, scaler=scale,
                               train=make_loader(train, 4, True), val=make_loader(val, 4),
                               audit={"synthetic": True})

    def config(self, epochs=1, model="dlinear"):
        return dict(model=model, dataset="ETTh1", seq_len=16, pred_len=8,
                    d_model=16, epochs=epochs, patience=3, seed=22,
                    lr=1e-3, weight_decay=1e-4, amp=False)

    def test_one_epoch_and_complete_reuse(self):
        with tempfile.TemporaryDirectory() as root:
            result = train_model(self.config(), self.bundle(), root, fingerprint="stable")
            self.assertTrue(result["complete"])
            self.assertEqual(result["epochs_completed"], 1)
            self.assertEqual(result["history"][0]["training_batches"], 3)
            stamp = (Path(root) / "best.pt").stat().st_mtime_ns
            repeated = train_model(self.config(), self.bundle(), root, fingerprint="stable")
            self.assertEqual(repeated, result)
            self.assertEqual(stamp, (Path(root) / "best.pt").stat().st_mtime_ns)
            state = load_checkpoint(Path(root) / "last.pt")
            for key in ("model", "optimizer", "scheduler", "rng", "history", "best", "stale", "epoch"):
                self.assertIn(key, state)

    def test_fingerprint_and_config_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as root:
            train_model(self.config(), self.bundle(), root, fingerprint="v1")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train_model(self.config(), self.bundle(), root, fingerprint="v2")
            config = self.config(); config["lr"] = 2e-3
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train_model(config, self.bundle(), root, fingerprint="v1")

    def test_interrupted_epoch_resume_matches_uninterrupted(self):
        config = self.config(epochs=2, model="tqnet")
        original = training.evaluate_model
        calls = [0]
        def interrupted(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 2:
                raise RuntimeError("simulated interruption")
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            with mock.patch.object(training, "evaluate_model", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    train_model(config, self.bundle(), path / "resumed", fingerprint="same")
            self.assertFalse((path / "resumed" / "complete.json").exists())
            result = train_model(config, self.bundle(), path / "resumed", fingerprint="same")
            uninterrupted = train_model(config, self.bundle(), path / "fresh", fingerprint="same")
            a = load_checkpoint(path / "resumed" / "last.pt")["model"]
            b = load_checkpoint(path / "fresh" / "last.pt")["model"]
            for name in a:
                torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
            self.assertEqual(result["best_score"], uninterrupted["best_score"])

    def test_nonfinite_training_fails_without_complete_marker(self):
        class BadModel(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.ones(()))
            def forward(self, x, **kwargs):
                return SimpleNamespace(mean=self.weight * torch.full((len(x), 8, 3), float("nan")),
                                       std=torch.ones(len(x), 8, 3))
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(training, "build_model", return_value=BadModel()):
                with self.assertRaises(FloatingPointError):
                    train_model(self.config(), self.bundle(), root)
            self.assertFalse((Path(root) / "complete.json").exists())


if __name__ == "__main__":
    unittest.main()
