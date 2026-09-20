FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    YUE_DEVICE=cuda YUE_BACKEND=torch YUE_QUANT=none YUE_MEMORY_BUDGET_GIB=24
RUN apt-get update && apt-get install -y --no-install-recommends \
    git g++ libsndfile1 ffmpeg ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip install "git+https://github.com/multimodal-art-projection/YuE.git@bd90e4ccae671d869b3ecaca6d7e893927d29442" \
    cog==0.21.0 runpod==1.7.9
ENV YUE_OFFLOAD_AR=0
WORKDIR /src
COPY predict.py yue_runtime.py weight_cache.py audio_transport.py rp_handler.py entry.sh cog.yaml ./
EXPOSE 5000
ENTRYPOINT ["/src/entry.sh"]
