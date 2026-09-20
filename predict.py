"""Cog predictor for YuE2 (m-a-p/YuE2-3B) lyrics-to-song generation.

Precision: upstream forces bfloat16 and disables TF32 for quality; fp16 is
not offered. Backend "torch" (default) uses CUDA graphs; "vllm" and fp8
quantization are env opt-ins. Weights fetch at runtime, cached on
/runpod-volume/hf when present. Env: YUE_MODEL, YUE_VAE, YUE_BACKEND,
YUE_QUANT, YUE_MEMORY_BUDGET_GIB (24), YUE_OFFLOAD_AR (1),
YUE_ODE_STEPS (32), YUE_DEVICE (auto cuda/cpu).
"""

import base64
import io
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict, Optional

_HF_HOME = os.environ.get("YUE_HF_HOME") or ("/runpod-volume/hf" if os.path.isdir("/runpod-volume") else "")
if _HF_HOME:
    os.makedirs(_HF_HOME, exist_ok=True)
    os.environ["HF_HOME"] = _HF_HOME

import numpy as np  # noqa: E402
import torch  # noqa: E402
from cog import BasePredictor, Input, Path  # noqa: E402

MODEL_ID = os.environ.get("YUE_MODEL", "m-a-p/YuE2-3B")
VAE_ID = os.environ.get("YUE_VAE", "m-a-p/YuE2-Vae")
BACKEND = os.environ.get("YUE_BACKEND", "torch")
QUANT = os.environ.get("YUE_QUANT", "none")
MEMORY_BUDGET = float(os.environ.get("YUE_MEMORY_BUDGET_GIB", "24"))
OFFLOAD_AR = os.environ.get("YUE_OFFLOAD_AR", "1") == "1"
DEVICE = os.environ.get("YUE_DEVICE", "")

FORMATS = ("flac", "wav", "mp3")


def ensure_weights() -> None:
    from huggingface_hub import snapshot_download

    t0 = time.time()
    for repo in (MODEL_ID, VAE_ID):
        snapshot_download(repo, local_files_only=False)
    print(f"[weights] ready in {time.time() - t0:.1f}s", flush=True)


def _write_audio(audio: np.ndarray, sample_rate: int, fmt: str, out_path: str) -> None:
    import soundfile as sf

    if fmt in ("flac", "wav"):
        sf.write(out_path, audio, sample_rate, subtype="PCM_24" if fmt == "flac" else "FLOAT")
        return
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sample_rate, subtype="FLOAT")
        wav_path = tmp.name
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav_path, "-codec:a", "libmp3lame",
                        "-q:a", "3", out_path], check=True)
    finally:
        os.unlink(wav_path)


def predict_defaults() -> Dict[str, object]:
    return {"cot": "full", "seed": 831001, "ode_steps": 32, "format": "flac"}


class Predictor(BasePredictor):
    def setup(self) -> None:
        ensure_weights()
        from yue2 import YuE2Pipeline
        from yue2.protocol import GenerationConfig

        gen = GenerationConfig(ode_steps=int(os.environ.get("YUE_ODE_STEPS", "32")))
        device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.pipe = YuE2Pipeline.from_pretrained(
            MODEL_ID, vae=VAE_ID, device=device,
            backend=BACKEND if device == "cuda" else "torch-eager",
            quantization=QUANT if device == "cuda" else "none",
            memory_budget_gib=MEMORY_BUDGET, offload_ar=OFFLOAD_AR,
            generation_config=gen, progress=False)
        self.sr = 48000

    def predict(
        self,
        style: str = Input(description="Genre, instruments, vocal character, language, tempo"),
        lyrics: str = Input(description="Lyrics with [Verse]/[Chorus] section tags"),
        cot: str = Input(description="Symbolic planning: full melody+chords, melody only, or off",
                         default="full", choices=["full", "melody", "off"]),
        seed: int = Input(description="Random seed", default=831001, ge=0),
        abc: str = Input(description="Optional own ABC score (requires full/melody)", default=""),
        cfg_scale: float = Input(description="Text guidance, blank for default", default=0, ge=0, le=20),
        ode_steps: int = Input(description="Flow-matching steps; 32 quality default, 16 fast", default=32, ge=1, le=128),
        semantic_max_tokens: int = Input(description="Cap song length for tests; 9000 full song", default=9000, ge=200, le=9000),
        format: str = Input(description="Delivery audio format", default="flac", choices=["flac", "wav", "mp3"]),
    ) -> Path:
        import dataclasses

        t0 = time.time()
        call = dict(style=style, lyrics=lyrics, cot=cot, seed=seed)
        if abc.strip():
            call["abc"] = abc
        if cfg_scale > 0:
            call["cfg_scale"] = cfg_scale
        old, self.pipe.generation_config = self.pipe.generation_config, dataclasses.replace(
            self.pipe.generation_config, ode_steps=ode_steps)
        try:
            song = self.pipe(semantic_sampling={"max_tokens": semantic_max_tokens}, **call)
        finally:
            self.pipe.generation_config = old
        ext = format if format in FORMATS else "flac"
        out_path = f"/tmp/yue2-{seed}-{int(time.time())}.{ext}"
        with torch.inference_mode():
            _write_audio(song.audio, song.sample_rate, ext, out_path)
        dt = time.time() - t0
        print(f"[predict] {len(song.audio) / song.sample_rate:.1f}s audio in {dt:.1f}s truncated={song.truncated}", flush=True)
        return Path(out_path)


def predict_audio_base64(inp: dict) -> dict:
    """Shared core for the RunPod handler: returns audio_b64 + format + timings."""
    from yue2 import YuE2Pipeline  # noqa: F401 (ensures import errors surface here)

    pred = _handler_predictor()
    out = pred.predict(
        style=inp.get("style", ""), lyrics=inp.get("lyrics", ""),
        cot=inp.get("cot", "full"), seed=int(inp.get("seed", 831001)),
        abc=inp.get("abc", ""), cfg_scale=float(inp.get("cfg_scale", 0)),
        ode_steps=int(inp.get("ode_steps", 32)),
        semantic_max_tokens=int(inp.get("semantic_max_tokens", 9000)),
        format=inp.get("format", "mp3") if inp.get("format") in FORMATS else "mp3")
    raw = open(str(out), "rb").read()
    fmt = str(out).rsplit(".", 1)[-1]
    return {"audio_b64": base64.b64encode(raw).decode(), "format": fmt,
            "bytes": len(raw)}


_handler_singleton: Optional[Predictor] = None


def _handler_predictor() -> Predictor:
    global _handler_singleton
    if _handler_singleton is None:
        _handler_singleton = Predictor()
        _handler_singleton.setup()
    return _handler_singleton
