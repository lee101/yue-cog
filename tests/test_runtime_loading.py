import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yue_runtime import Runtime


class RuntimeLoadingTests(unittest.TestCase):
    def run_load(self, outcomes):
        pipeline = Mock()
        pipeline.from_pretrained.side_effect = outcomes
        missing = type("LocalEntryNotFoundError", (Exception,), {})
        modules = {"torch": Mock(), "yue2": types.SimpleNamespace(YuE2Pipeline=pipeline),
                   "yue2.protocol": types.SimpleNamespace(GenerationConfig=Mock()),
                   "huggingface_hub.errors": types.SimpleNamespace(LocalEntryNotFoundError=missing)}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"YUE_DEVICE": "cpu"}), \
                patch("weight_cache.install_verified_hash_cache"):
            Runtime().load()
        return pipeline.from_pretrained.call_args_list

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
