from __future__ import annotations

import base64
import dataclasses
import math
import os
from pathlib import Path
import tempfile
import threading
import time

HF_HOME = os.getenv("YUE_HF_HOME") or ("/runpod-volume/hf" if os.path.isdir("/runpod-volume") else "")
if HF_HOME:
    os.environ["HF_HOME"] = HF_HOME

MODEL_REVISION = "14fc6c6f146441b1dd6363fcb2e01e82a6914cb7"
VAE_REVISION = "9a94e1d0ea9f8087e98f77fa88df4a4068104d2a"
RUNTIME_VERSION = "yue2-bd90e4c-quality-v2"


def normalize_request(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("input must be an object")
    out = {}
    for key, limit in (("style", 2000), ("lyrics", 12000), ("abc", 16000)):
        value = raw.get(key, "")
        if not isinstance(value, str) or len(value) > limit:
            raise ValueError(f"{key} must be text up to {limit} characters")
        out[key] = value.strip()
    if not out["style"] or not out["lyrics"]:
        raise ValueError("style and lyrics are required")
    for key, default, low, high in (("seed", 831001, 0, 2147483647),
                                   ("ode_steps", 32, 16, 64),
                                   ("semantic_max_tokens", 9000, 200, 9000)):
        value = raw.get(key, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}")
        out[key] = value
    for key, default, allowed in (("cot", "full", ("full", "melody", "off")),
                                  ("format", "flac", ("flac", "mp3", "wav"))):
        value = raw.get(key, default)
        if value not in allowed:
            raise ValueError(f"invalid {key}")
        out[key] = value
    value = raw.get("cfg_scale", 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 20:
        raise ValueError("cfg_scale must be between 0 and 20")
    out["cfg_scale"] = value
    if out["abc"] and out["cot"] == "off":
        raise ValueError("abc requires full or melody planning")
    return out


class Runtime:
    def __init__(self) -> None:
        self.pipe = None
        self.lock = threading.Lock()

    def load(self) -> None:
        import torch
        from yue2 import YuE2Pipeline
        from yue2.protocol import GenerationConfig
        from huggingface_hub.errors import LocalEntryNotFoundError
        from weight_cache import install_verified_hash_cache

        install_verified_hash_cache()

        device = os.getenv("YUE_DEVICE", "cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU generation must be explicitly selected")
        if device == "cpu":
            torch.set_num_threads(int(os.getenv("YUE_CPU_THREADS", "8")))
        options = dict(
            vae=os.getenv("YUE_VAE", "m-a-p/YuE2-Vae"),
            revision=os.getenv("YUE_MODEL_REVISION", MODEL_REVISION),
            vae_revision=os.getenv("YUE_VAE_REVISION", VAE_REVISION),
            device=device,
            backend=os.getenv("YUE_BACKEND", "torch") if device == "cuda" else "torch-eager",
            quantization=os.getenv("YUE_QUANT", "none") if device == "cuda" else "none",
            memory_budget_gib=float(os.getenv("YUE_MEMORY_BUDGET_GIB", "18")),
            offload_ar=os.getenv("YUE_OFFLOAD_AR", "1") == "1",
            generation_config=GenerationConfig(ode_steps=32), progress=False)
        model = os.getenv("YUE_MODEL", "m-a-p/YuE2-3B")
        pinned = all(len(options[key]) == 40 and all(c in "0123456789abcdef" for c in options[key])
                     for key in ("revision", "vae_revision"))
        if pinned:
            try:
                self.pipe = YuE2Pipeline.from_pretrained(model, local_files_only=True, **options)
                return
            except (LocalEntryNotFoundError, FileNotFoundError):
                pass
        self.pipe = YuE2Pipeline.from_pretrained(model, **options)

    def generate(self, raw: dict) -> dict:
        import numpy as np
        import soundfile as sf
        import subprocess
        import torch

        request = normalize_request(raw)
        started = time.monotonic()
        with self.lock, torch.inference_mode():
            if os.getenv("YUE_DEVICE", "cuda") == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            if self.pipe is None:
                self.load()
            previous = self.pipe.generation_config
            self.pipe.generation_config = dataclasses.replace(previous, ode_steps=request["ode_steps"])
            call = {k: request[k] for k in ("style", "lyrics", "cot", "seed")}
            for key in ("abc", "cfg_scale"):
                if request[key]:
                    call[key] = request[key]
            try:
                song = self.pipe(semantic_sampling={"max_tokens": request["semantic_max_tokens"]}, **call)
            finally:
                self.pipe.generation_config = previous
            if len(song.audio) == 0 or not np.isfinite(song.audio).all():
                raise RuntimeError("model produced invalid audio")
            fmt = request["format"]
            with tempfile.TemporaryDirectory(prefix="yue2-") as directory:
                path = Path(directory) / f"song.{fmt}"
                if fmt == "mp3":
                    wav = Path(directory) / "song.wav"
                    sf.write(wav, song.audio, song.sample_rate, subtype="FLOAT")
                    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(wav),
                                    "-codec:a", "libmp3lame", "-q:a", "2", str(path)],
                                   check=True, timeout=120)
                else:
                    sf.write(path, song.audio, song.sample_rate,
                             subtype="PCM_24" if fmt == "flac" else "FLOAT")
                audio = path.read_bytes()
            return {"audio_b64": base64.b64encode(audio).decode(), "format": fmt,
                    "content_type": {"mp3": "audio/mpeg", "flac": "audio/flac", "wav": "audio/wav"}[fmt],
                    "bytes": len(audio), "seconds": len(song.audio) / song.sample_rate,
                    "sample_rate": song.sample_rate, "truncated": song.truncated,
                    "semantic_tokens": len(song.semantic.tokens),
                    "device": str(self.pipe.device), "runtime_backend": self.pipe.backend,
                    "quantization": self.pipe.quantization,
                    "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)
                        if self.pipe.device.type == "cuda" else 0,
                    "wall_s": round(time.monotonic() - started, 3),
                    "timing": song.timing, "runtime": RUNTIME_VERSION}
