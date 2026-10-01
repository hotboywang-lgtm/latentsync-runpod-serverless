FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface \
    CACHE_ROOT=/runpod-volume/latentsync-cache \
    OUTPUT_MODE=s3

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl ffmpeg git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 https://github.com/bytedance/LatentSync.git /app/LatentSync
WORKDIR /app/LatentSync

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && pip install runpod boto3 filelock

# Bake every runtime weight into the image so cold workers do not download
# models after a request has entered the queue.
RUN mkdir -p checkpoints/whisper checkpoints/sd-vae-ft-mse \
    && huggingface-cli download ByteDance/LatentSync-1.6 \
      whisper/tiny.pt latentsync_unet.pt --local-dir checkpoints \
    && huggingface-cli download stabilityai/sd-vae-ft-mse \
      --local-dir checkpoints/sd-vae-ft-mse

COPY handler.py /app/LatentSync/handler.py
COPY README.md /app/LatentSync/RUNPOD_README.md

CMD ["python", "-u", "handler.py"]
