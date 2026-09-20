import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weight_cache import verified_digest


class WeightCacheTests(unittest.TestCase):
    def test_huggingface_snapshot_symlink_reuses_blob_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / "unextended-blob-hash"
            blob.write_bytes(b"weights")
            snapshot = root / "model.safetensors"
            snapshot.symlink_to(blob)
            hasher = Mock(return_value="a" * 64)
            self.assertEqual(verified_digest(snapshot, hasher, root / "cache"), "a" * 64)
            self.assertEqual(verified_digest(snapshot, hasher, root / "cache"), "a" * 64)
            hasher.assert_called_once_with(blob)

    def test_reuses_only_unchanged_verified_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = root / "model.safetensors"
            weight.write_bytes(b"original")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            hasher = Mock(side_effect=digest)
            first = verified_digest(weight, hasher, root / "cache")
            self.assertEqual(first, verified_digest(weight, hasher, root / "cache"))
            self.assertEqual(hasher.call_count, 1)
            stat = weight.stat()
            weight.write_bytes(b"modified")
            os.utime(weight, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertNotEqual(first, verified_digest(weight, hasher, root / "cache"))
            self.assertEqual(hasher.call_count, 2)

    def test_replacement_and_corrupt_cache_are_reverified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = root / "model.safetensors"
            weight.write_bytes(b"first")
            hasher = Mock(side_effect=lambda path: hashlib.sha256(path.read_bytes()).hexdigest())
            verified_digest(weight, hasher, root / "cache")
            next((root / "cache").glob("*.json")).write_text("invalid")
            verified_digest(weight, hasher, root / "cache")
            weight.unlink()
            weight.write_bytes(b"other")
            verified_digest(weight, hasher, root / "cache")
            self.assertEqual(hasher.call_count, 3)

    def test_mid_read_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = root / "model.safetensors"
            weight.write_bytes(b"first")
            def hasher(path):
                path.write_bytes(b"changed")
                return "0" * 64
            with self.assertRaises(RuntimeError):
                verified_digest(weight, hasher, root / "cache")


if __name__ == "__main__":
    unittest.main()
