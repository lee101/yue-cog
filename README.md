# YuE2 song generation Cog (app.nz + RunPod serverless)

Lyrics + style in, 48kHz stereo song out. Upstream: [YuE2](https://github.com/multimodal-art-projection/YuE) (`m-a-p/YuE2-3B` + `m-a-p/YuE2-Vae`).

## Precision

Upstream forces **bfloat16** and disables TF32 for quality; fp16 is deliberately not offered (H3-style fp16 shortcuts cost audible quality here). Accel comes from CUDA-graph backend (`torch`, default), bf16 AR + flow-matching, tiled VAE decode, and a persistent warm pipeline. Opt-ins: `YUE_BACKEND=vllm`, `YUE_QUANT=fp8`.

## Layout

- `predict.py` — Cog predictor (`style`, `lyrics`, `cot`, `seed`, `abc`, `cfg_scale`, `ode_steps`, `semantic_max_tokens`, `format`)
- `rp_handler.py` — RunPod serverless handler, returns `audio_b64` + `format`
- `cog.yaml` / `Dockerfile.sls` / `entry.sh` — dual-mode image (cog HTTP / runpod)
- `appnz.schema.json` — app.nz deploy schema (`outputKind: audio`)
- `scripts/runpod_deploy.py` — endpoint create/update, defaults **10s scaledown, zero workers until explicitly enabled**
- `scripts/runpod_smoke.py` — one song end-to-end
- `scripts/local_gen.py` — local GPU test via `.venv`
- `tools/c-bench/` — C audio-postprocess microbench

## Local

```
python3 -m venv .venv && .venv/bin/pip install "git+https://github.com/multimodal-art-projection/YuE.git@bd90e4ccae671d869b3ecaca6d7e893927d29442"
.venv/bin/python scripts/apply_yue2_patch.py          # exact accel patch, see below
.venv/bin/python scripts/local_gen.py --fast --out /tmp/yue2-test.flac
```

`local_gen.py` auto-selects device: CUDA only when free VRAM covers the
model (~14 GiB bf16, ~7.5 GiB fp8) plus 1.5 GiB headroom, else CPU.
Full song (9000 tokens, ode 32) needs a quiet 24 GB GPU; on a shared box
use `--fast` (1500 tokens, ode 16) or cap `--tokens 3000 --ode-steps 32`.
CPU (16 threads, no AVX512) decodes ~7 tok/s: a 1500-token clip takes
tens of minutes with the same BF16 quality preset. Same-seed outputs can differ
between CPU and GPU; the recorded 4090 cold/warm pair was byte-identical.

Speed levers, quality-safe first: `semantic_max_tokens` (linear in song
length) > `ode_steps` 32→16 (fewer NAR evaluations, requires quality comparison) >
`cot` full→melody (skips chord planning). `YUE_QUANT=fp8` halves AR
VRAM but is experimental with no quality claim; `YUE_BACKEND=vllm`
needs vllm installed and serves one request at a time. `YUE_OFFLOAD_AR=1`
(default) cuts NAR peak VRAM at small time cost.

## Accel patch (exact)

`patches/yue2-accel.patch` modifies the pinned upstream package; `scripts/apply_yue2_patch.py`
applies it to the interpreter's `yue2` (idempotent, refuses anything but the pinned pristine
files, `--check`, `--revert`). The Dockerfile applies it at build. `cog.yaml` builds stay
unpatched and `yue_runtime.py` then runs the upstream paths. Every change produces the same
tokens, latents and FLAC bytes as upstream, so `RUNTIME_VERSION` and result caches are unchanged.

| Env (default) | Effect |
| --- | --- |
| `YUE_FAST_PLACEMENT` (1) | Weights keep an immutable CPU master (the loaded mmap): offload is free, AR is not re-uploaded after NAR only to be offloaded for the VAE. One upload per request instead of two uploads and two downloads. Same peak VRAM. |
| `YUE_FUSED_SAMPLING` (1) | Single-branch requests run logits processing, the multinomial draw and history update inside the decode CUDA graph; one host sync per token instead of several round-trips. Verified bitwise against the eager sampler once per process, else falls back. CFG/`cot=off` keep eager sampling. |
| `YUE_LEAN_DECODE` (1) | Q/K heads share one RMSNorm+RoPE pass per layer (fewer tiny kernels). Verified bitwise (logits and KV) once per process, else falls back. |
| `YUE_RESIDENT` (0, image 1) | LM and VAE decoder stay on the GPU; no per-request moves. Peak allocation 9.6 GiB instead of 7.6 GiB (LM plus VAE tile), idle process VRAM 8.4 GiB instead of 1.5 GiB. Needs the patch. |
| `YUE_PIN_WEIGHTS` (0) | Pinned masters: upload 0.5 s instead of 1.9 s, for about 8 s once per process and about 9 GiB locked host RAM. Only worth it for long-lived non-resident workers. |

`scripts/accel_ab.py` interleaves legacy/accel/resident renders in one process and exits
non-zero unless tokens, latents, audio and encoded bytes are identical; run it on a new GPU
type before trusting the timings.

RTX 3090 Ti, 68.2 s clip (Electric Gold, seed 831002, 3000-token cap, 32 steps), warm, min / median
of 3, GPU otherwise idle: upstream paths 39.20 / 39.57 s, defaults 29.26 / 29.38 s,
`YUE_RESIDENT=1` 27.65 / 28.54 s. AR decode 106 -> 133 tok/s. With another tenant holding the
GPU at 100%: 77.4 / 78.0 s -> 63.8 s. All 40 patched renders matched the unpatched FLAC byte for
byte. Stage tables, placement costs and rejected ideas: `examples/accel-ab-20261110.json`.

## Deploy

```
scripts/build_push_deploy.sh [endpoint-id]   # needs RUNPOD_API_KEY, HF_TOKEN
RUNPOD_API_KEY=... scripts/runpod_smoke.py --endpoint <id> --style "..." --lyrics "[Verse] ..."
```

Weights (~7GB) fetch at runtime into `HF_HOME` (`/runpod-volume/hf` on serverless).

## License

Adapter code MIT. YuE2 model weights: CC BY-NC 4.0; we hold a commercial license from the YuE authors for production use. Publicly the service is the "Manifold Music Generator".

## Native serving and NetWRCK

`yue_runtime.py` is shared by Cog, RunPod and OmniServe's worker. It pins model
and VAE revisions, validates bounded requests, serializes inference, uses full
inference mode, and returns duration, truncation and stage timings. Serverless
encoding uses request-scoped temporary files and removes them on completion.
`YUE_DEVICE=cuda` fails clearly when CUDA is unavailable; CPU is explicit.
Verified weight digests are reused only while file identity and timestamps
match; upstream manifest verification remains enabled. Set
`YUE_WEIGHT_HASH_CACHE` to a private service-owned cache directory if needed.

Build the dual-mode runtime with `docker build --network host -t IMAGE .`.
This uses the CUDA 12.8 PyTorch wheel on a slim Python base, avoiding a full
CUDA development base. The original Cog configuration remains available.
RunPod deployment defaults to zero workers; pass `--workers-max 1` explicitly
for a test, then reset it to zero. The inference image defaults to a 24 GiB
budget on a dedicated worker; the shared-host service uses 18 GiB.

Native configuration and cost/quality details are in
`../omniserve-native/performance/yue2-music.md`. Production: ManifoldGen's Manifold Music Generator (server/music_yue.go) submits to RunPod endpoint omniserve-yue2-quality; nothing gates generation. `examples/electric-gold.json` contains
the original prompt and public bucket URLs for the default quality sample.

Install `requirements-serving.txt` in the caller/worker virtualenv to sign R2
uploads. Native RunPod overflow and the benchmark/smoke scripts require the R2
environment variables documented in the native performance guide. The remote
container receives a presigned PUT for one object, uploads lossless audio
directly to the bucket, and returns a small URL/metadata result. Bucket secrets
stay on the submitting host. Direct callers may omit `output_upload` only for
inline results under 6 MiB.
