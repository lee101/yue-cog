#!/usr/bin/env python3
"""Submit one song job to the RunPod endpoint and print timings.

  RUNPOD_API_KEY=... scripts/runpod_smoke.py --endpoint <id> \
      --style "English, warm piano pop, ..." --lyrics "[Verse] ..." --out /tmp/smoke.mp3
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request

BASE = "https://api.runpod.ai/v2"


def call(api_key, method, url, body=None, retries=8):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key,
                                          "User-Agent": "yue-cog/0.1"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"retry {attempt + 1}: {exc}", file=sys.stderr, flush=True)
            time.sleep(5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--lyrics", required=True)
    ap.add_argument("--cot", default="full")
    ap.add_argument("--seed", type=int, default=831001)
    ap.add_argument("--ode-steps", type=int, default=32)
    ap.add_argument("--semantic-max-tokens", type=int, default=9000)
    ap.add_argument("--format", default="mp3")
    ap.add_argument("--out", default="/tmp/yue2-smoke.mp3")
    args = ap.parse_args()
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        sys.exit("RUNPOD_API_KEY is required")
    payload = {"input": {"style": args.style, "lyrics": args.lyrics, "cot": args.cot,
                         "seed": args.seed, "ode_steps": args.ode_steps,
                         "semantic_max_tokens": args.semantic_max_tokens, "format": args.format},
               "policy": {"executionTimeout": 1800000}}
    t0 = time.time()
    job = call(api_key, "POST", f"{BASE}/{args.endpoint}/run", payload)
    job_id = job["id"]
    print("job", job_id, job.get("status"), flush=True)
    while True:
        time.sleep(5)
        st = call(api_key, "GET", f"{BASE}/{args.endpoint}/status/{job_id}")
        if st.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            out = st.get("output") or {}
            if out.get("audio_b64"):
                raw = base64.b64decode(out["audio_b64"])
                open(args.out, "wb").write(raw)
                print(f"wrote {args.out} ({len(raw)} bytes)")
                out = {k: (f"<{len(v)} chars>" if k == "audio_b64" else v) for k, v in out.items()}
            print(json.dumps({"status": st.get("status"), "error": st.get("error"),
                              "executionTime_ms": st.get("executionTime"),
                              "wall_s": round(time.time() - t0, 1), "output": out}, indent=1))
            sys.exit(0 if st.get("status") == "COMPLETED" else 1)


if __name__ == "__main__":
    main()
