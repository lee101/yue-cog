import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yue_runtime import Runtime


class RuntimeLoadingTests(unittest.TestCase):
    def run_load(self, outcomes, env=None, **package):
        pipeline = Mock()
        pipeline.from_pretrained.side_effect = outcomes
        missing = type("LocalEntryNotFoundError", (Exception,), {})
        modules = {"torch": Mock(), "yue2": types.SimpleNamespace(YuE2Pipeline=pipeline, **package),
                   "yue2.protocol": types.SimpleNamespace(GenerationConfig=Mock()),
                   "huggingface_hub.errors": types.SimpleNamespace(LocalEntryNotFoundError=missing)}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"YUE_DEVICE": "cpu", **(env or {})}), \
                patch("weight_cache.install_verified_hash_cache"):
            Runtime().load()
        return pipeline.from_pretrained.call_args_list

    def test_unpatched_package_gets_no_accel_options(self):
        kwargs = self.run_load([Mock()])[0].kwargs
        self.assertFalse({"resident", "fast_placement", "fused_sampling", "lean_decode", "pin_weights"} & kwargs.keys())
        with self.assertRaisesRegex(RuntimeError, "apply_yue2_patch"):
            self.run_load([Mock()], env={"YUE_RESIDENT": "1"})

    def test_patched_package_defaults_are_exact_and_vram_neutral(self):
        kwargs = self.run_load([Mock()], ACCEL_PATCH=1)[0].kwargs
        self.assertEqual([kwargs[key] for key in ("resident", "pin_weights", "fast_placement", "fused_sampling", "lean_decode")],
                         [False, False, True, True, True])
        kwargs = self.run_load([Mock()], env={"YUE_DEVICE": "cuda", "YUE_RESIDENT": "1", "YUE_FUSED_SAMPLING": "0"},
                               ACCEL_PATCH=1)[0].kwargs
        self.assertEqual((kwargs["resident"], kwargs["fused_sampling"]), (True, False))

    def test_pinned_cached_revision_requires_no_network(self):
        calls = self.run_load([Mock()])
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].kwargs["local_files_only"])

    def test_missing_snapshot_downloads_pinned_revision(self):
        calls = self.run_load([FileNotFoundError(), Mock()])
        self.assertEqual(len(calls), 2)
        self.assertNotIn("local_files_only", calls[1].kwargs)
        self.assertEqual(calls[0].kwargs["revision"], calls[1].kwargs["revision"])

    def test_integrity_error_is_never_bypassed(self):
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.run_load([ValueError("weight integrity failed"), Mock()])


if __name__ == "__main__":
    unittest.main()
