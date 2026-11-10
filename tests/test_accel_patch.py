import importlib.util
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("apply_yue2_patch", ROOT / "scripts/apply_yue2_patch.py")
apply_patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apply_patch)


class AccelPatchTests(unittest.TestCase):
    def test_patch_only_touches_verified_files(self):
        text = apply_patch.PATCH.read_text()
        touched = set(re.findall(r"^\+\+\+ b/yue2/(\S+)", text, re.M))
        self.assertEqual(touched, set(apply_patch.PRISTINE) | {"placement.py"})
        self.assertNotIn("modeling_vae.py", touched)
        for value in apply_patch.PRISTINE.values():
            self.assertRegex(value, r"^[0-9a-f]{64}$")

    def test_defaults_keep_release_semantics(self):
        text = apply_patch.PATCH.read_text()
        self.assertIn("resident=False, fast_placement=True, fused_sampling=True, lean_decode=True, pin_weights=False", text)
        for check in ("_verify_sampler", "_verify_lean"):
            self.assertIn(check, text)


if __name__ == "__main__":
    unittest.main()
