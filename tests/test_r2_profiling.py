"""Tests focus on evidence integrity, state restoration and honest failure rows."""
import json
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from kamo_r2.profiling import profile_model, profile_sweep


class TinyForecaster(nn.Module):
    def __init__(self, seq_len=8, horizon=3):
        super().__init__()
        self.linear = nn.Linear(seq_len, horizon)
        self.dropout = nn.Dropout(0.1)
        self.raw_std = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("calls", torch.tensor(0.0))

    def forward(self, x, mask=None):
        if self.training:
            self.calls += 1
        mean = self.linear(self.dropout(x).transpose(1, 2)).transpose(1, 2)
        return SimpleNamespace(mean=mean, std=torch.nn.functional.softplus(self.raw_std).expand_as(mean), auxiliary_loss=None)


class ProfilingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_restores_weights_buffers_gradients_modes_and_rng(self):
        model = TinyForecaster().train()
        model.dropout.eval()  # Preserve mixed module modes too.
        model.raw_std.grad = torch.tensor(3.0)
        x = torch.randn(2, 8, 2)
        original = {key: value.clone() for key, value in model.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        report = profile_model(model, x, warmup=1, repeats=2, train_warmup=1,
                               train_repeats=2, measure_flops=False)
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(float(model.raw_std.grad), 3.0)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[name]), name)
        self.assertGreater(report["inference"]["median_ms"], 0)
        self.assertGreater(report["train_step"]["median_ms"], 0)
        self.assertIsNone(report["inference"]["peak_allocated_bytes"])
        json.dumps(report, allow_nan=False)

    def test_flops_are_partial_or_missing_never_fabricated_total(self):
        report = profile_model(TinyForecaster(), torch.randn(2, 8, 2),
                               warmup=0, repeats=1, train_repeats=0)
        flops = report["flops"]
        self.assertIn(flops["status"], {"partial_count", "unavailable", "no_supported_operator_counts"})
        self.assertIn("incomplete", flops["coverage"])
        if flops["counted_flops_per_batch"] is not None:
            self.assertGreater(flops["counted_flops_per_batch"], 0)
        json.dumps(report, allow_nan=False)

    def test_state_restored_when_training_callback_fails(self):
        model = TinyForecaster().eval()
        original = {key: value.clone() for key, value in model.state_dict().items()}
        x = torch.randn(2, 8, 2)

        def bad_loss(*args):
            raise RuntimeError("intentional callback failure")

        with self.assertRaisesRegex(RuntimeError, "intentional callback failure"):
            profile_model(model, x, warmup=0, repeats=1, train_repeats=1,
                          measure_flops=False, loss_fn=bad_loss)
        self.assertFalse(model.training)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[name]))

    def test_sweep_builds_each_case_and_does_not_shrink_after_oom(self):
        visits = []

        def build(case):
            visits.append(case["variables"])
            if case["variables"] == 4:
                raise RuntimeError("CUDA out of memory")
            x = torch.zeros(2, 8, case["variables"])
            return TinyForecaster(), x, torch.ones_like(x, dtype=torch.bool)

        rows = profile_sweep(build, [{"variables": 2}, {"variables": 4}, {"variables": 6}],
                             expected_batch_size=2, warmup=0, repeats=1,
                             train_repeats=0, measure_flops=False)
        self.assertEqual(visits, [2, 4, 6])
        self.assertEqual([row["result"]["status"] for row in rows], ["ok", "oom", "ok"])
        self.assertEqual(rows[1]["result"]["batch_size_policy"], "unchanged_no_automatic_reduction")

    def test_invalid_inputs_and_batch_changes_fail_explicitly(self):
        with self.assertRaises(ValueError):
            profile_model(TinyForecaster(), torch.zeros(2, 8), repeats=1)
        rows = profile_sweep(lambda case: (TinyForecaster(), torch.zeros(1, 8, 2), None),
                             [{}], expected_batch_size=2, repeats=1)
        self.assertEqual(rows[0]["result"]["status"], "error")
        self.assertIn("common batch size", rows[0]["result"]["error"])

    def test_cycle_index_forward_kwargs_used_during_all_phases(self):
        class PhasedModel(TinyForecaster):
            def forward(self, x, mask=None, cycle_index=None):
                if cycle_index is None:
                    raise ValueError("missing real cycle phase")
                self.asserted_phase = cycle_index.detach().cpu().tolist()
                return super().forward(x, mask)

        model = PhasedModel()
        report = profile_model(model, torch.randn(2, 8, 2), warmup=1, repeats=1,
                               train_warmup=1, train_repeats=1, measure_flops=True,
                               forward_kwargs={"cycle_index": torch.tensor([17, 18])})
        self.assertEqual(model.asserted_phase, [17, 18])
        self.assertEqual(report["status"], "ok")


if __name__ == "__main__":
    unittest.main()
