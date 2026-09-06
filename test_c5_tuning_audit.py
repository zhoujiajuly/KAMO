import argparse
import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
tempfile.tempdir = str(HERE / "_test_tmp")
Path(tempfile.tempdir).mkdir(exist_ok=True)
SPEC = importlib.util.spec_from_file_location("c5", HERE / "run_c5_tuning_audit.py")
c5 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c5)


def cli(cmd):
    out = {}
    i = 3
    while i < len(cmd):
        key = cmd[i]
        if key == "--skip-test":
            out["skip_test"] = True
            i += 1
        elif key.startswith("--"):
            if key == "--conditions":
                j = i + 1
                values = []
                while j < len(cmd) and not cmd[j].startswith("--"):
                    values.append(cmd[j])
                    j += 1
                out["conditions"] = values
                i = j
            else:
                out[key[2:].replace("-", "_")] = cmd[i + 1]
                i += 2
        else:
            i += 1
    return out


class FakeBackend:
    def __init__(self):
        self.train_calls = []
        self.validation_calls = []
        self.test_calls = []

    def __call__(self, cmd, log, dry=False):
        if dry:
            return
        values = cli(cmd)
        script = cmd[2]
        if script == "train.py":
            self.train_calls.append(list(cmd))
            dataset = values["dataset"]
            model = values["model"]
            seed = int(values["seed"])
            run_name = values["run_name"]
            spec = next(s for s in c5.SPECS if ("c5_" + s + "_") in run_name)
            chosen = None
            for settings in c5.candidates(dataset, spec):
                if (int(values["d_model"]) == settings["d_model"]
                        and abs(float(values["lr"]) - settings["lr"]) < 1e-12
                        and abs(float(values["dropout"]) - settings["dropout"]) < 1e-12):
                    chosen = settings
                    break
            if chosen is None:
                # Final seeds use the selected settings; they still belong to
                # one of the frozen candidates.
                raise AssertionError("Command is outside the frozen candidate grid")
            run = Path(values["output_root"]) / dataset / model / run_name
            run.mkdir(parents=True)
            config = c5.expected_config(dataset, spec, chosen, seed)
            config.update(output_root=values["output_root"], data_root=values["data_root"], run_name=run_name)
            (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (run / "best.pt").write_bytes(("%s|%s|%s|%s" % (dataset, spec, seed, chosen["candidate"])).encode())
            (run / "scaler.npz").write_bytes(b"fake-unit-test-scaler")
            (run / "history.csv").write_text("epoch,val_crps\n1,1.0\n", encoding="utf-8")
            (run / "selection.json").write_text(json.dumps(dict(
                dataset=dataset, model=model, seed=seed, selection_split="validation",
                selection_metric="crps", best_val_crps=1.0, best_epoch=1, test_accessed=False
            )), encoding="utf-8")
            return
        if script != "evaluate_robustness.py":
            raise AssertionError("Unexpected command: " + str(cmd))
        run = Path(values["run_dir"])
        config = c5.read_json(run / "config.json")
        split = values["split"]
        conditions = values["conditions"]
        output = Path(values["output"])
        if split == "validation":
            self.validation_calls.append(list(cmd))
            self.assert_validation_grid(conditions)
        else:
            self.test_calls.append(list(cmd))
        offset = 0.1
        name = run.name
        if "c0_current" in name:
            offset = 0.2
        elif "c1_lr_half" in name:
            offset = 0.1
        elif "c2_lr_double" in name:
            offset = 0.3
        elif "c3_dropout_0p2" in name:
            offset = 0.4
        offset += (int(config["seed"]) - 2026) * 0.01
        fields = ["dataset", "model", "seed", "condition", "parameters", "evaluation_split",
                  "full_evaluation", "full_test", "calibration_type", "calibration_applied",
                  "mae", "crps", "nll", "picp_95", "mpiw_95", "interval_score_95"]
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for i, condition in enumerate(conditions):
                score = offset + i * 0.001
                writer.writerow(dict(
                    dataset=config["dataset"], model=config["model"], seed=config["seed"],
                    condition=condition, parameters=int(config["d_model"]) * 1000 + 33,
                    evaluation_split=split, full_evaluation=True, full_test=split == "test",
                    calibration_type="none", calibration_applied=False,
                    mae=score + 0.1, crps=score, nll=score + 1,
                    picp_95=0.95, mpiw_95=score + 2, interval_score_95=score + 3))
    
    @staticmethod
    def assert_validation_grid(conditions):
        if tuple(conditions) != c5.VALIDATION:
            raise AssertionError("Search accessed a non-frozen or unseen condition")


class Tests(unittest.TestCase):
    def make_args(self, root):
        root = Path(root)
        code = root / "code"
        data = root / "data"
        code.mkdir()
        data.mkdir()
        for relative in c5.SOURCE_FILES:
            path = code / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            # Command-surface tests use the real parser sources copied below.
            source = ""
            real = HERE.parent.parent.parent / "0830KAMO_soft" / relative
            if real.is_file():
                source = real.read_text(encoding="utf-8")
            path.write_text(source, encoding="utf-8")
        (data / "ETTh2.csv").write_text("date,a\n0,1\n", encoding="utf-8")
        (data / "weather.csv").write_text("date,a\n0,1\n", encoding="utf-8")
        return argparse.Namespace(
            code_root=str(code), data_root=str(data),
            search_root=str(root / "search"), final_root=str(root / "final"),
            results_root=str(root / "results"), device="cuda", python="python3",
            lane=0, dry_run=False,
        )

    def freeze(self, args):
        c5.write_exact(Path(args.results_root) / "FROZEN_PROTOCOL.json", c5.protocol(args))

    def test_case_partition_and_counts(self):
        cases = c5.LANES[0] + c5.LANES[1]
        self.assertEqual(len(cases), 14)
        self.assertEqual(len(set(cases)), 14)
        self.assertEqual(set(cases), {(d, s) for d in c5.DATASETS for s in c5.SPECS})
        self.assertEqual(sum(len(c5.candidates(d, s)) for d, s in cases), 56)

    def test_candidate_grid_changes_one_axis(self):
        for dataset in c5.DATASETS:
            for spec in c5.SPECS:
                values = c5.candidates(dataset, spec)
                self.assertEqual([v["candidate"] for v in values],
                                 ["c0_current", "c1_lr_half", "c2_lr_double", "c3_dropout_0p2"])
                self.assertEqual(values[3]["lr"], values[0]["lr"])
                self.assertEqual(values[3]["dropout"], 0.2)
                for key in ("d_model", "patch_len", "n_layers", "vp_kernel_vars"):
                    self.assertEqual(len({v[key] for v in values}), 1)

    def test_capacity_configurations_are_not_core_aliases(self):
        self.assertEqual(c5.candidates("ETTh2", "timemixer_core")[0]["d_model"], 64)
        self.assertEqual(c5.candidates("ETTh2", "timemixer_capacity")[0]["d_model"], 512)
        self.assertEqual(c5.SPECS["robust_patchtst_capacity"][0], "patchtst_robust")
        self.assertEqual(c5.candidates("weather", "robust_patchtst_capacity")[0]["d_model"], 128)

    def test_search_and_capacity_paths_cannot_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.make_args(tmp)
            core = c5.paths(args, "weather", "timemixer_core",
                            c5.candidates("weather", "timemixer_core")[0], 2026, True)
            cap = c5.paths(args, "weather", "timemixer_capacity",
                           c5.candidates("weather", "timemixer_capacity")[0], 2026, True)
            self.assertNotEqual(core, cap)

    def test_protocol_binds_sources_and_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.make_args(tmp)
            before = c5.protocol(args)
            Path(args.data_root, "ETTh2.csv").write_text("changed", encoding="utf-8")
            after = c5.protocol(args)
            self.assertNotEqual(before["dataset_sha256"], after["dataset_sha256"])
            Path(args.code_root, "train.py").write_text("changed", encoding="utf-8")
            self.assertNotEqual(after["source_sha256"], c5.protocol(args)["source_sha256"])

    def test_write_exact_refuses_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen.json"
            c5.write_exact(path, {"a": 1})
            c5.write_exact(path, {"a": 1})
            with self.assertRaises(RuntimeError):
                c5.write_exact(path, {"a": 2})

    def test_source_change_after_freeze_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.make_args(tmp)
            self.freeze(args)
            c5.assert_code_frozen(args)
            Path(args.code_root, "kamo", "model.py").write_text("changed", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                c5.assert_code_frozen(args)

    def test_train_command_uses_skip_test_and_supported_flags(self):
        real_train = HERE / "train.py"
        if not real_train.is_file():
            real_train = HERE.parent.parent.parent / "0830KAMO_soft" / "train.py"
        if not real_train.is_file():
            self.skipTest("Local source mirror not present")
        text = real_train.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            args = self.make_args(tmp)
            settings = c5.candidates("ETTh2", "kamo_core")[0]
            run, _ = c5.paths(args, "ETTh2", "kamo_core", settings, 2026, True)
            command = c5.train_command(args, "ETTh2", "kamo_core", settings, 2026, run)
        self.assertIn("--skip-test", command)
        self.assertNotIn("--max-test-batches", command)
        for value in command:
            if value.startswith("--"):
                self.assertIn('"' + value + '"', text)

    def test_csv_validator_rejects_test_marked_as_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.csv"
            fields = ["dataset", "model", "seed", "condition", "parameters", "evaluation_split",
                      "full_evaluation", "full_test", "calibration_type", "calibration_applied", *c5.METRICS]
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for condition in c5.VALIDATION:
                    w.writerow(dict(dataset="ETTh2", model="patchtst", seed=2026, condition=condition,
                                    parameters=10, evaluation_split="validation", full_evaluation=True,
                                    full_test=True, calibration_type="none", calibration_applied=False,
                                    mae=1, crps=1, nll=1, picp_95=.95, mpiw_95=1, interval_score_95=1))
            with self.assertRaises(RuntimeError):
                c5.validate_csv(path, "ETTh2", "patchtst", 2026, c5.VALIDATION, "validation")

    def test_incomplete_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.make_args(tmp)
            self.freeze(args)
            settings = c5.candidates("ETTh2", "kamo_core")[0]
            run, _ = c5.paths(args, "ETTh2", "kamo_core", settings, 2026, True)
            run.mkdir(parents=True)
            (run / "marker").write_text("user evidence", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                c5.ensure_training(args, "ETTh2", "kamo_core", settings, 2026, run)
            self.assertTrue((run / "marker").is_file())

    def test_end_to_end_search_final_resume_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.make_args(tmp)
            self.freeze(args)
            backend = FakeBackend()
            with patch.object(c5, "stream", side_effect=backend):
                for dataset in c5.DATASETS:
                    for spec in c5.SPECS:
                        c5.search_case(args, dataset, spec)
                self.assertEqual(len(backend.train_calls), 56)
                self.assertEqual(len(backend.validation_calls), 56)
                self.assertEqual(len(backend.test_calls), 0)
                selected = c5.all_search_complete(args)
                self.assertTrue(all(x["selected"]["candidate"] == "c1_lr_half" for x in selected.values()))
                # Search is perfectly resumable and does not rerun completed evidence.
                for dataset in c5.DATASETS:
                    for spec in c5.SPECS:
                        c5.search_case(args, dataset, spec)
                self.assertEqual(len(backend.train_calls), 56)
                self.assertEqual(len(backend.validation_calls), 56)
                for dataset in c5.DATASETS:
                    for spec in c5.SPECS:
                        c5.final_case(args, dataset, spec)
                self.assertEqual(len(backend.train_calls), 84)
                self.assertEqual(len(backend.test_calls), 42)
                c5.summarize(args)
                audit = c5.read_json(Path(args.results_root) / "FINAL_AUDIT.json")
                self.assertTrue(audit["complete"])
                self.assertEqual(audit["search_candidates"], 56)
                self.assertEqual(audit["selected_configurations"], 14)
                self.assertEqual(audit["condition_rows"], 462)
                # Final phase is also resumable without rerunning training/evaluation.
                for dataset in c5.DATASETS:
                    for spec in c5.SPECS:
                        c5.final_case(args, dataset, spec)
                self.assertEqual(len(backend.train_calls), 84)
                self.assertEqual(len(backend.test_calls), 42)

    def test_tampered_winner_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.make_args(tmp)
            self.freeze(args)
            backend = FakeBackend()
            with patch.object(c5, "stream", side_effect=backend):
                c5.search_case(args, "ETTh2", "kamo_core")
            path = c5.selection_path(args, "ETTh2", "kamo_core")
            obj = c5.read_json(path)
            obj["selected"] = obj["candidates"][-1]
            path.write_text(json.dumps(obj), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                c5.validate_selection(args, "ETTh2", "kamo_core")

    def test_final_requires_all_fourteen_selections(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.make_args(tmp)
            self.freeze(args)
            with self.assertRaises(RuntimeError):
                c5.all_search_complete(args)

    def test_lane_lock_blocks_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "lane.lock"
            with c5.LaneLock(lock):
                with self.assertRaises(RuntimeError):
                    with c5.LaneLock(lock):
                        pass
            self.assertFalse(lock.exists())


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        unittest.main(verbosity=2)
