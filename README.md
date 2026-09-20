# YuE2 song generation Cog (app.nz + RunPod serverless)

Lyrics + style in, 48kHz stereo song out. Upstream: [YuE2](https://github.com/multimodal-art-projection/YuE) (`m-a-p/YuE2-3B` + `m-a-p/YuE2-Vae`).

## Precision

Upstream forces **bfloat16** and disables TF32 for quality; fp16 is deliberately not offered (H3-style fp16 shortcuts cost audible quality here). Accel comes from CUDA-graph backend (`torch`, default), bf16 AR + flow-matching, tiled VAE decode, and a persistent warm pipeline. Opt-ins: `YUE_BACKEND=vllm`, `YUE_QUANT=fp8`.

## Layout

- `predict.py` — Cog predictor (`style`, `lyrics`, `cot`, `seed`, `abc`, `cfg_scale`, `ode_steps`, `semantic_max_tokens`, `format`)
- `rp_handler.py` — RunPod serverless handler, returns `audio_b64` + `format`
- `cog.yaml` / `Dockerfile.sls` / `entry.sh` — dual-mode image (cog HTTP / runpod)
- `appnz.schema.json` — app.nz deploy schema (`outputKind: audio`)
- `scripts/runpod_deploy.py` — endpoint create/update, defaults **10s scaledown, max 1 worker**
- `scripts/runpod_smoke.py` — one song end-to-end
- `scripts/local_gen.py` — local GPU test via `.venv`
- `tools/c-bench/` — C audio-postprocess microbench

## Local

```
python3 -m venv .venv && .venv/bin/pip install "git+https://github.com/multimodal-art-projection/YuE.git@bd90e4ccae671d869b3ecaca6d7e893927d29442"
.venv/bin/python scripts/local_gen.py --fast --out /tmp/yue2-test.flac
```

`local_gen.py` auto-selects device: CUDA only when free VRAM covers the
model (~14 GiB bf16, ~7.5 GiB fp8) plus 1.5 GiB headroom, else CPU.
Full song (9000 tokens, ode 32) needs a quiet 24 GB GPU; on a shared box
use `--fast` (1500 tokens, ode 16) or cap `--tokens 3000 --ode-steps 32`.
CPU (16 threads, no AVX512) decodes ~7 tok/s: a 1500-token clip takes
tens of minutes, quality identical to GPU (same bf16 preset).

Speed levers, quality-safe first: `semantic_max_tokens` (linear in song
length) > `ode_steps` 32→16 (~2x NAR, slight flow-matching cost) >
`cot` full→melody (skips chord planning). `YUE_QUANT=fp8` halves AR
VRAM but is experimental with no quality claim; `YUE_BACKEND=vllm`
needs vllm installed and serves one request at a time. `YUE_OFFLOAD_AR=1`
(default) cuts NAR peak VRAM at small time cost.

## Deploy

```
scripts/build_push_deploy.sh [endpoint-id]   # needs RUNPOD_API_KEY, HF_TOKEN
RUNPOD_API_KEY=... scripts/runpod_smoke.py --endpoint <id> --style "..." --lyrics "[Verse] ..."
```

Weights (~7GB) fetch at runtime into `HF_HOME` (`/runpod-volume/hf` on serverless).

## License

Adapter code MIT. YuE2 model weights: CC BY-NC 4.0 + creator permission — commercial use needs a license from the YuE authors; generated audio is free to monetize for personal/creator use.
