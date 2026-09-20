#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import time
import urllib.request
import urllib.error
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_transport import prepare_output_upload


def call(url: str, method: str = "GET", payload: dict | None = None) -> dict:
    request = urllib.request.Request(url, method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"],
                 "Content-Type": "application/json", "User-Agent": "Omniserve-YuE/1"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read(2048).decode(errors="replace")
        print(f"RunPod HTTP {error.code}: {detail}", flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--runs", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    control = "https://rest.runpod.io/v1/endpoints/" + args.endpoint
    base = "https://api.runpod.ai/v2/" + args.endpoint
    config = call(control)
    if config.get("name") != "omniserve-yue2-quality" or config.get("workersMax") != 0:
        raise SystemExit("benchmark requires the dedicated, disabled omniserve-yue2-quality endpoint")
    manifest = json.loads((Path(__file__).resolve().parents[1] / "examples/electric-gold.json").read_text())
    request = {key: manifest[key] for key in ("style", "lyrics", "seed", "cot", "ode_steps")}
    request.update(semantic_max_tokens=3000, format="flac")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    active = None
    results = []
    try:
        call(control, "PATCH", {"workersMin": 0, "workersMax": 1, "idleTimeout": 60})
        time.sleep(5)
        for index in range(args.runs):
            start = time.monotonic()
            target = prepare_output_upload(request["format"])
            job = call(base + "/run", "POST", {"input": {**request, "output_upload": target}, "policy": {"executionTimeout": 480000, "ttl": 1200000}})
            active = job["id"]
            print(f"run={index + 1} job={active}", flush=True)
            deadline = start + 1200
            previous_status = None
            while time.monotonic() < deadline:
                state = call(base + "/status/" + active)
                status = state.get("status")
                if status != previous_status:
                    print(f"run={index + 1} status={status} elapsed_s={time.monotonic() - start:.1f}", flush=True)
                    previous_status = status
                if status == "COMPLETED":
                    result = state.get("output") or {}
                    if result.get("error") or not (result.get("audio_b64") or result.get("audio_url")):
                        raise RuntimeError("worker returned an error: " + str(result.get("error")))
                    if result.get("audio_url"):
                        download = urllib.request.Request(result["audio_url"], headers={"User-Agent": "Omniserve-YuE/1"})
                        with urllib.request.urlopen(download, timeout=120) as response:
                            audio = response.read()
                    else:
                        audio = base64.b64decode(result.pop("audio_b64"), validate=True)
                    (output / f"run-{index + 1}.flac").write_bytes(audio)
                    result.update(execution_ms=state.get("executionTime"), delay_ms=state.get("delayTime"),
                                  request_wall_s=round(time.monotonic() - start, 3), job_id=active,
                                  worker_id=state.get("workerId"), request=request)
                    results.append(result)
                    (output / "results.json").write_text(json.dumps(results, indent=2))
                    print(json.dumps(result), flush=True)
                    active = None
                    break
                if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                    raise RuntimeError("benchmark job " + status + ": " + str(state.get("error", "")))
                time.sleep(5)
            else:
                raise TimeoutError("benchmark job timed out")
    finally:
        try:
            if active:
                call(base + "/cancel/" + active, "POST", {})
        finally:
            call(control, "PATCH", {"workersMin": 0, "workersMax": 0, "idleTimeout": 10})
            final = call(control)
            print(json.dumps({"endpoint_id": args.endpoint, "workersMin": final.get("workersMin"),
                              "workersMax": final.get("workersMax")}), flush=True)


if __name__ == "__main__":
    main()
