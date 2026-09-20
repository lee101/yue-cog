import base64
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_transport import deliver_audio


class AudioTransportTests(unittest.TestCase):
    def test_upload_preserves_audio_and_returns_small_metadata(self):
        output = {"audio_b64": base64.b64encode(b"audio").decode(), "content_type": "audio/flac", "seconds": 60}
        target = {"put_url": "https://test.r2.cloudflarestorage.com/bucket/audio?signature=example",
                  "audio_url": "https://cdn.example/audio.flac", "content_type": "audio/flac"}
        response = MagicMock()
        response.__enter__.return_value.status = 200
        with patch("audio_transport.urllib.request.urlopen", return_value=response) as upload:
            result = deliver_audio(output, target)
            self.assertEqual(upload.call_args.args[0].data, b"audio")
            self.assertEqual(upload.call_args.args[0].method, "PUT")
        self.assertEqual(result, {"audio_url": target["audio_url"], "content_type": "audio/flac", "seconds": 60})

    def test_large_inline_and_invalid_targets_fail(self):
        with self.assertRaises(RuntimeError):
            deliver_audio({"audio_b64": "a" * (6 * 1024**2 + 1)}, None)
        with self.assertRaises(ValueError):
            deliver_audio({"audio_b64": "", "content_type": "audio/flac"},
                          {"put_url": "http://127.0.0.1/private", "audio_url": "https://cdn.example/a"})


if __name__ == "__main__":
    unittest.main()
