import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path("branches/trajectory_audit_lab/scripts/relocate_experiment_outputs.py").resolve()
SPEC = importlib.util.spec_from_file_location("relocate_experiment_outputs", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


class RelocateExperimentOutputsTests(unittest.TestCase):
    def test_classify_results_dir(self):
        self.assertEqual(MOD.classify_results_dir("audit_cli"), "audit_history")
        self.assertEqual(MOD.classify_results_dir("_smoke_tracks_new"), "test_artifacts")
        self.assertEqual(MOD.classify_results_dir("figures"), "audit_history")
        self.assertIsNone(MOD.classify_results_dir("results_Kepler-432_20260228_194343"))

    def test_build_move_plans_filters_expected_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            root.mkdir()
            (root / "audit_alpha").mkdir()
            (root / "_smoke_one").mkdir()
            (root / "results_foo").mkdir()

            branch = Path(tmp) / "branches" / "trajectory_audit_lab"
            plans = MOD.build_move_plans(root, branch)

            mapped = {(p.source.name, p.category) for p in plans}
            self.assertIn(("audit_alpha", "audit_history"), mapped)
            self.assertIn(("_smoke_one", "test_artifacts"), mapped)
            self.assertNotIn(("results_foo", "audit_history"), mapped)


if __name__ == "__main__":
    unittest.main()
