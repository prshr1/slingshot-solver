import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path("branches/trajectory_audit_lab/scripts/trajectory_gradient_renderer.py").resolve()
SPEC = importlib.util.spec_from_file_location("trajectory_gradient_renderer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


class TrajectoryGradientRendererTests(unittest.TestCase):
    def test_find_latest_run_dir_picks_newest_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            d1 = root / "run_a"
            d1.mkdir()
            (d1 / "results.pkl").write_bytes(b"0")
            (d1 / "config.yaml").write_text("system: {}\n", encoding="utf-8")

            d2 = root / "run_b"
            d2.mkdir()
            (d2 / "results.pkl").write_bytes(b"0")
            (d2 / "config.yaml").write_text("system: {}\n", encoding="utf-8")

            # Invalidate one candidate to ensure filtering works.
            d3 = root / "run_c"
            d3.mkdir()
            (d3 / "results.pkl").write_bytes(b"0")

            d1.touch()
            d2.touch()
            d3.touch()

            # Force mtime ordering.
            d1_m = d1.stat().st_mtime - 100
            d2_m = d1.stat().st_mtime + 100
            import os

            os.utime(d1, (d1_m, d1_m))
            os.utime(d2, (d2_m, d2_m))

            found = MOD.find_latest_run_dir(root)
            self.assertEqual(found.name, "run_b")

    def test_resolve_output_dir_uses_tag_or_run_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "out"
            run_dir = Path(tmp) / "results_foo"
            run_dir.mkdir()

            out1 = MOD.resolve_output_dir(out_root, run_dir, None)
            self.assertTrue(out1.exists())
            self.assertEqual(out1.name, "results_foo")

            out2 = MOD.resolve_output_dir(out_root, run_dir, "custom_tag")
            self.assertTrue(out2.exists())
            self.assertEqual(out2.name, "custom_tag")


if __name__ == "__main__":
    unittest.main()
