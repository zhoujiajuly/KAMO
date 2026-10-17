import unittest
from kamo_r2.protocol import candidates, recipe_cases, case_shard, case_name
from run_r2 import conditions_for, make_plan, parse_args, case_protocol
from types import SimpleNamespace
from kamo_r2.storage import digest


class ProtocolTests(unittest.TestCase):
    def test_effective_distinct_candidates(self):
        for dataset, dimensions in [("ETTh2", 7), ("exchange", 8), ("weather", 21)]:
            rows = candidates("kamo_s", dataset, 12, num_vars=dimensions)
            signatures = [tuple(sorted(row.items())) for row in rows]
            self.assertEqual(len(set(signatures)), 12)
            self.assertTrue(all(row["mixer_rank"] <= dimensions for row in rows))
        linear = candidates("dlinear", "ETTh2", 12)
        self.assertEqual(len(set((r["lr"], r["weight_decay"]) for r in linear)), 12)
        self.assertTrue(all(not r["linear_individual"] for r in linear))

    def test_shards_are_disjoint_complete(self):
        cases = recipe_cases("core")
        left = {case_name(c) for c in cases if case_shard(c, 2) == 0}
        right = {case_name(c) for c in cases if case_shard(c, 2) == 1}
        self.assertFalse(left & right)
        self.assertEqual(len(left | right), len(cases))

    def test_control_parent_protocol_matches_core(self):
        args = parse_args(["--recipe", "core"])
        case = next(c for c in recipe_cases("core") if c["model"] == "kamo_s")
        bundle = SimpleNamespace(audit={"data_sha256": "example"})
        core = case_protocol(args, case, bundle, "source")
        args.recipe = "controls"
        parent = dict(recipe_cases("controls")[0], model="kamo_s", inherited=False)
        control = case_protocol(args, parent, bundle, "source")
        self.assertEqual(digest(core), digest(control))

    def test_condition_aliases_are_not_double_counted(self):
        conditions = conditions_for("all")
        self.assertEqual(conditions.count("level_shift:1"), 1)
        self.assertNotIn("level_shift:1.0", conditions)
        self.assertEqual(len(set(conditions)), len(conditions))


if __name__ == "__main__":
    unittest.main()
