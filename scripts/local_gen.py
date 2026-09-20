#!/usr/bin/env python3
import argparse
import dataclasses
import json
import os
import time

DEFAULT_STYLE = ("English, warm piano pop, expressive female voice, acoustic piano, "
                 "rounded bass and light drums, lyrical memorable melody, 88 BPM")
DEFAULT_LYRICS = ("[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\n"
                  "[Chorus]\nLet the day come into view\nEvery road begins with you")


def pick_device(args):
    import torch
    if args.device != "auto":
        return args.device
    if not torch.cuda.is_available():
        return "cpu"
    free, _ = torch.cuda.mem_get_info()
    need = 7.5 if args.quant == "fp8" else 14.0
    if free / 2**30 < need + 1.5:
        print(f"cuda free {free/2**30:.1f}GiB < {need + 1.5:.1f} needed, using cpu", flush=True)
        return "cpu"
    return "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/yue2-local.flac")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--model", default=os.environ.get("YUE_MODEL", "m-a-p/YuE2-3B"))
    ap.add_argument("--vae", default=os.environ.get("YUE_VAE", "m-a-p/YuE2-Vae"))
    ap.add_argument("--style", default=DEFAULT_STYLE)
    ap.add_argument("--lyrics", default=DEFAULT_LYRICS)
    ap.add_argument("--seed", type=int, default=831001)
    ap.add_argument("--cot", default="full", choices=["full", "melody", "off"])
    ap.add_argument("--ode-steps", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=0)
    ap.add_argument("--backend", default=os.environ.get("YUE_BACKEND", "torch"),
                    choices=["torch", "torch-eager", "vllm"])
    ap.add_argument("--quant", default=os.environ.get("YUE_QUANT", "none"), choices=["none", "fp8"])
    ap.add_argument("--budget", type=float, default=float(os.environ.get("YUE_MEMORY_BUDGET_GIB", "0")))
    ap.add_argument("--offload", default=os.environ.get("YUE_OFFLOAD_AR", "1") == "1",
                    action=argparse.BooleanOptionalAction)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    ode = args.ode_steps or (16 if args.fast else 32)
    tokens = args.tokens or (1500 if args.fast else 9000)

    import torch
    from yue2 import YuE2Pipeline
    from yue2.protocol import GenerationConfig

    device = pick_device(args)
    if device == "cpu":
        torch.set_num_threads(args.threads)
        backend, quant = "torch-eager", "none"
    else:
        backend, quant = args.backend, args.quant
    budget = args.budget
    if device == "cuda" and not budget:
        free, _ = torch.cuda.mem_get_info()
        budget = max(4.0, min(free / 2**30 - 1.5, 24.0))
    elif not budget:
        budget = 24.0
    print(f"device={device} backend={backend} quant={quant} budget={budget:.1f} ode={ode} tokens={tokens}",
          flush=True)

    t0 = time.time()
    pipe = YuE2Pipeline.from_pretrained(args.model, vae=args.vae, device=device,
        backend=backend, quantization=quant, memory_budget_gib=budget,
        offload_ar=args.offload, generation_config=GenerationConfig(ode_steps=ode),
        progress=False)
    print(f"load {time.time() - t0:.1f}s", flush=True)
    t1 = time.time()
    song = pipe(style=args.style, lyrics=args.lyrics, cot=args.cot, seed=args.seed,
                semantic_sampling={"max_tokens": tokens})
    print(f"gen {time.time() - t1:.1f}s truncated={song.truncated}", flush=True)
    song.save(args.out)
    print(json.dumps({"audio": args.out, "seconds": round(len(song.audio) / song.sample_rate, 1),
                      "sem_tokens": len(song.semantic.tokens), "wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
