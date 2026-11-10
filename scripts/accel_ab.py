#!/usr/bin/env python3
"""Same-process A/B of the yue2 accel patch on this GPU: stage timing plus exactness.

  .venv/bin/python scripts/accel_ab.py --rounds 3 --out /tmp/yue-accel-ab.json
Exit status 1 when any accelerated variant differs from the legacy path in tokens, latents or audio.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OFF = dict(fast_placement=False, fused_sampling=False, lean_decode=False, resident=False, pin_weights=False)
ON = dict(OFF, fast_placement=True, fused_sampling=True, lean_decode=True)
VARIANTS = {"legacy": OFF, "accel": ON, "accel+pin": dict(ON, pin_weights=True), "accel+resident": dict(ON, resident=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=3000)
    ap.add_argument("--variants", default="legacy,accel,accel+resident")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    import numpy as np
    import yue2.pipeline
    from yue_runtime import Runtime

    request = json.loads((Path(__file__).resolve().parents[1] / "examples/gpu-benchmark-20260920.json").read_text())
    request = {**request["results"][0]["request"], "semantic_max_tokens": args.tokens}
    last = {}
    original = yue2.pipeline.YuE2Pipeline.__call__

    def call(self, *a, **k):
        last["song"] = original(self, *a, **k)
        return last["song"]
    yue2.pipeline.YuE2Pipeline.__call__ = call
    runtime = Runtime()
    runtime.load()
    if not runtime.accel:
        raise SystemExit("yue2 is unpatched; run scripts/apply_yue2_patch.py")
    offload = runtime.pipe.offload_ar
    rows, reference, exact = [], None, True
    for index in range(args.rounds):
        for name in args.variants.split(","):
            for key, value in VARIANTS[name].items():
                setattr(runtime.pipe, key, value)
            runtime.pipe.offload_ar = offload and not VARIANTS[name]["resident"]
            output = runtime.generate(dict(request))
            song, timing = last["song"], output["timing"]
            identity = [hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for value in (
                np.asarray(song.semantic.plan.abc_ids), np.asarray(song.semantic.tokens), song.latents, song.audio)]
            identity.append(hashlib.sha256(output["audio_b64"].encode()).hexdigest())
            reference = reference or identity
            same = identity == reference
            exact &= same
            stages = dict(wall=output["wall_s"], abc=timing["abc"]["seconds"], semantic=timing["semantic"]["seconds"],
                          nar=timing["nar_seconds"], vae=timing["vae_seconds"])
            stages["other"] = stages["wall"] - sum(stages[k] for k in ("abc", "semantic", "nar", "vae"))
            rows.append(dict(variant=name, round=index, identical=same, peak_allocated_mib=output["peak_allocated_mib"],
                             abc_tps=timing["abc"]["output_tps"], semantic_tps=timing["semantic"]["output_tps"],
                             seconds=output["seconds"], **stages))
            print(json.dumps(rows[-1]), flush=True)
    summary = {}
    for name in args.variants.split(","):
        own = [row for row in rows if row["variant"] == name]
        summary[name] = {key: {"min": round(min(row[key] for row in own), 3),
                               "median": round(statistics.median(row[key] for row in own), 3)}
                         for key in ("wall", "abc", "semantic", "nar", "vae", "other", "peak_allocated_mib")}
    import torch
    report = dict(gpu=torch.cuda.get_device_name(0), offload_ar=offload, request=request, exact=exact,
                  summary=summary, rows=rows)
    print(json.dumps({"exact": exact, "summary": summary}, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    return 0 if exact else 1


if __name__ == "__main__":
    sys.exit(main())
