import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "deploy", Path(__file__).resolve().parents[1] / "scripts/runpod_deploy.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class DeployTests(unittest.TestCase):
    def test_zero_is_explicitly_patched(self):
        with patch.object(deploy, "api", side_effect=[
            {"id": "test", "workersMax": 3},
            {"id": "test", "workersMin": 0, "workersMax": 0},
        ]) as api:
            self.assertEqual(deploy.save_endpoint("key", "test", "template", "ADA_24", 0, 10), "test")
            self.assertEqual(api.call_args.args,
                             ("key", "PATCH", "/endpoints/test", {"workersMin": 0, "workersMax": 0}))

    def test_worker_limit_mismatch_fails(self):
        with patch.object(deploy, "api", side_effect=[
            {"id": "test"}, {"id": "test", "workersMin": 0, "workersMax": 3},
        ]):
            with self.assertRaises(RuntimeError):
                deploy.save_endpoint("key", "test", "template", "ADA_24", 0, 10)


if __name__ == "__main__":
    unittest.main()
