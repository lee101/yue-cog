"""RunPod serverless handler. Keeps the shared YuE2 runtime warm between jobs."""

import os
import sys
import threading

import runpod
from yue_runtime import Runtime, normalize_request
from audio_transport import deliver_audio

runtime = Runtime()


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
    if not isinstance(payload, dict) or not payload.get("style") or not payload.get("lyrics"):
        return {"error": "style and lyrics are required"}
    try:
        result = runtime.generate(normalize_request(payload))
        return deliver_audio(result, payload.get("output_upload"))
    except Exception as exc:  # noqa: BLE001
        _recover_gpu(str(exc))
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    print(f"[worker] starting pod={os.environ.get('RUNPOD_POD_ID')} endpoint={os.environ.get('RUNPOD_ENDPOINT_ID')}", flush=True)
    runpod.serverless.start({"handler": handler})
