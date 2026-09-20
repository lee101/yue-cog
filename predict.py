"""Cog interface to the shared quality-preserving YuE2 runtime."""

import base64
import tempfile
import threading
from typing import Optional

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        from yue_runtime import Runtime
        self.runtime = Runtime()
        self.runtime.load()

    def predict(
        self,
        style: str = Input(description="Genre, instruments, vocal character, language, tempo"),
        lyrics: str = Input(description="Lyrics with [Verse]/[Chorus] section tags"),
        cot: str = Input(description="Symbolic planning: full melody+chords, melody only, or off",
                         default="full", choices=["full", "melody", "off"]),
        seed: int = Input(description="Random seed", default=831001, ge=0, le=2147483647),
        abc: str = Input(description="Optional own ABC score (requires full/melody)", default=""),
        cfg_scale: float = Input(description="Text guidance, blank for default", default=0, ge=0, le=20),
        ode_steps: int = Input(description="Flow-matching steps; 32 quality default, 16 fast", default=32, ge=16, le=64),
        semantic_max_tokens: int = Input(description="Cap song length for tests; 9000 full song", default=9000, ge=200, le=9000),
        format: str = Input(description="Delivery audio format", default="flac", choices=["flac", "wav", "mp3"]),
    ) -> Path:
        result = self.runtime.generate(dict(style=style, lyrics=lyrics, cot=cot, seed=seed,
            abc=abc, cfg_scale=cfg_scale, ode_steps=ode_steps,
            semantic_max_tokens=semantic_max_tokens, format=format))
        with tempfile.NamedTemporaryFile(prefix="yue2-", suffix="." + result["format"], delete=False) as out:
            out.write(base64.b64decode(result["audio_b64"]))
        return Path(out.name)


def predict_audio_base64(inp: dict) -> dict:
    return _handler_predictor().runtime.generate(inp)


_handler_singleton: Optional[Predictor] = None
_handler_lock = threading.Lock()


def _handler_predictor() -> Predictor:
    global _handler_singleton
    with _handler_lock:
        if _handler_singleton is None:
            candidate = Predictor()
            candidate.setup()
            _handler_singleton = candidate
    return _handler_singleton
