"""RunPod serverless handler. Calls the cog Predictor in-process (no HTTP hop)."""

import os
import socket
import sys
import threading
import time

import runpod
from predict import _handler_predictor

LOG_PATH = "/tmp/yue2-worker.log"


def _recover_gpu(message):
    import torch

    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            if free < 0.15 * total:
                sys.__stderr__.write(f"[worker] GPU still full after job ({message}); exiting\n")
                threading.Timer(2.0, lambda: os._exit(3)).start()
    except Exception as exc:  # noqa: BLE001
        sys.__stderr__.write(f"[worker] gpu recover failed: {exc}\n")


def handler(job):
    payload = job.get("input") or {}
    if not payload.get("style") or not payload.get("lyrics"):
        return {"error": "style and lyrics are required"}
    t0 = time.time()
    try:
        pred = _handler_predictor()
        out = pred.predict(
            style=payload["style"], lyrics=payload["lyrics"],
            cot=payload.get("cot", "full"), seed=int(payload.get("seed", 831001)),
            abc=payload.get("abc", ""), cfg_scale=float(payload.get("cfg_scale", 0)),
            ode_steps=int(payload.get("ode_steps", 32)),
            semantic_max_tokens=int(payload.get("semantic_max_tokens", 9000)),
            format=payload.get("format", "mp3"))
        import base64
        raw = open(str(out), "rb").read()
        fmt = str(out).rsplit(".", 1)[-1]
        try:
            os.unlink(str(out))
        except OSError:
            pass
        return {"audio_b64": base64.b64encode(raw).decode(), "format": fmt,
                "bytes": len(raw), "wall_s": round(time.time() - t0, 1)}
    except Exception as exc:  # noqa: BLE001
        _recover_gpu(str(exc))
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    print(f"[worker] starting pod={os.environ.get('RUNPOD_POD_ID')} endpoint={os.environ.get('RUNPOD_ENDPOINT_ID')}", flush=True)
    runpod.serverless.start({"handler": handler})
