"""Fast CPU invariants and upstream-mean equivalence; no dataset downloads."""
import unittest
from types import SimpleNamespace

import torch

from kamo_r2.models import (NEW_MODEL_NAMES, OfficialGaussianForecaster,
                            build_new_model, new_model_search_space)


class TestNewBaselines(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.randn(3, 32, 5)
        self.starts = torch.tensor([0, 17, 40])

    def model(self, name, **kwargs):
        return OfficialGaussianForecaster(name, 5, 32, 12, d_model=16,
                                          cycle_len=24, **kwargs)

    def test_forward_backward_all(self):
        for name in NEW_MODEL_NAMES:
            with self.subTest(name=name):
                model = self.model(name)
                output = model(self.x, cycle_index=self.starts)
                self.assertEqual(output.mean.shape, (3, 12, 5))
                self.assertTrue(torch.isfinite(output.mean).all().item())
                self.assertTrue(torch.isfinite(output.std).all().item())
                self.assertTrue((output.std > 0).all().item())
                loss = output.std.log().mean() + (output.mean / output.std).square().mean()
                loss.backward()
                for p in model.parameters():
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all().item())

    def test_mean_identical_to_upstream(self):
        for name in NEW_MODEL_NAMES:
            with self.subTest(name=name):
                model = self.model(name).eval()
                with torch.no_grad():
                    expected = (model.backbone(self.x, self.starts % 24)
                                if name in ("cyclenet", "tqnet") else model.backbone(self.x))
                    actual = model(self.x, cycle_index=self.starts).mean
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_masked_values_do_not_change_prediction(self):
        mask = torch.ones_like(self.x, dtype=torch.bool)
        mask[:, 3:7, 0] = False
        mask[0, :, 2] = False
        bad_x = self.x.clone()
        bad_x[~mask] = float("nan")
        other_x = self.x.clone()
        other_x[~mask] = 1e10
        for name in NEW_MODEL_NAMES:
            with self.subTest(name=name):
                model = self.model(name).eval()
                a = model(bad_x, mask, cycle_index=self.starts)
                b = model(other_x, mask, cycle_index=self.starts)
                torch.testing.assert_close(a.mean, b.mean)
                torch.testing.assert_close(a.std, b.std)
                self.assertTrue(torch.isfinite(a.mean).all().item())

    def test_cycle_phase_required_and_order_independent(self):
        permutation = torch.tensor([2, 0, 1])
        for name in ("cyclenet", "tqnet"):
            model = self.model(name).eval()
            with self.assertRaisesRegex(ValueError, "global window starts"):
                model(self.x)
            with self.assertRaisesRegex(ValueError, "one start"):
                model(self.x, cycle_index=[0])
            a = model(self.x, cycle_index=self.starts).mean
            b = model(self.x[permutation], window_start=self.starts[permutation]).mean
            torch.testing.assert_close(b, a[permutation])

    def test_dlinear_has_separate_seasonal_and_trend(self):
        model = self.model("dlinear")
        self.assertTrue(hasattr(model.backbone, "Linear_Seasonal"))
        self.assertTrue(hasattr(model.backbone, "Linear_Trend"))
        seasonal, trend = model.backbone.decompsition(self.x)
        torch.testing.assert_close(seasonal + trend, self.x)
        expected = (model.backbone.Linear_Seasonal(seasonal.transpose(1, 2)) +
                    model.backbone.Linear_Trend(trend.transpose(1, 2))).transpose(1, 2)
        torch.testing.assert_close(model(self.x).mean, expected)

    def test_individual_linear_backward_and_factory(self):
        for name in ("dlinear", "nlinear"):
            args = SimpleNamespace(seq_len=32, pred_len=12, dataset="ETTh1",
                                   linear_individual=True)
            model = build_new_model(name, args, SimpleNamespace(num_vars=5))
            out = model(self.x)
            (out.mean.square().mean() + out.std.mean()).backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_search_parameters_are_effective(self):
        for name in ("dlinear", "nlinear"):
            space = new_model_search_space(name, "ETTh1")
            self.assertNotIn("dropout", space)
            self.assertNotIn("d_model", space)
        self.assertNotIn("dropout", new_model_search_space("cyclenet", "ETTh1"))
        self.assertEqual(new_model_search_space("tqnet", "exchange")["cycle_len"], [5, 20])


if __name__ == "__main__":
    unittest.main()
