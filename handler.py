import hashlib
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import boto3
import requests
import runpod
import torch
from botocore.config import Config
from DeepCache import DeepCacheSDHelper
from diffusers import AutoencoderKL, DDIMScheduler
from filelock import FileLock
from omegaconf import OmegaConf

from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature


APP_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = Path(os.getenv("CACHE_ROOT", "/runpod-volume/latentsync-cache"))
if str(CACHE_ROOT).startswith("/runpod-volume") and not Path("/runpod-volume").exists():
    CACHE_ROOT = Path("/tmp/latentsync-cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "sources").mkdir(parents=True, exist_ok=True)

MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", str(1024 * 1024 * 1024)))
DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", "120"))
PRESIGNED_URL_TTL = int(os.getenv("PRESIGNED_URL_TTL", "86400"))
OUTPUT_MODE = os.getenv("OUTPUT_MODE", "s3").lower()
MODEL_LOCK = FileLock(str(CACHE_ROOT / "latentsync-inference.lock"))


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _safe_id(value: str, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value):
        raise ValueError(f"{field} must contain only letters, numbers, '.', '_' or '-'")
    return value


def _bounded_int(payload: dict, name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(payload.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _bounded_float(payload: dict, name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(payload.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _download(url: str, destination: Path) -> tuple[str, int]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are accepted")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    with requests.get(url, stream=True, timeout=(15, DOWNLOAD_TIMEOUT), allow_redirects=True) as response:
        response.raise_for_status()
        length = response.headers.get("content-length")
        if length and int(length) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Remote file exceeds MAX_DOWNLOAD_BYTES")
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Remote file exceeds MAX_DOWNLOAD_BYTES")
                digest.update(chunk)
                handle.write(chunk)
    return digest.hexdigest(), total


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _normalize_video(source: Path, destination: Path, duration: float) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-vf",
            "fps=25,scale=trunc(iw/2)*2:trunc(ih/2)*2,setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )


def _normalize_audio(source: Path, destination: Path, duration: float) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )


def _finalize_video(source: Path, destination: Path) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )


def _s3_client():
    access_key = os.environ["S3_ACCESS_KEY_ID"]
    secret_key = os.environ["S3_SECRET_ACCESS_KEY"]
    signature_version = os.getenv("S3_SIGNATURE_VERSION", "s3v4")
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        region_name=os.getenv("S3_REGION", "auto"),
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version=signature_version),
    )


def _publish(path: Path, job_id: str) -> tuple[str, str]:
    if OUTPUT_MODE != "s3":
        raise RuntimeError("This endpoint requires OUTPUT_MODE=s3")
    bucket = os.environ["S3_BUCKET"]
    prefix = os.getenv("S3_PREFIX", "latentsync").strip("/")
    key = f"{prefix}/outputs/{job_id}-{uuid.uuid4().hex[:10]}.mp4"
    client = _s3_client()
    client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": "video/mp4"})
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=PRESIGNED_URL_TTL,
    )
    return key, url


def _load_pipeline():
    started = time.perf_counter()
    config = OmegaConf.load(APP_ROOT / "configs/unet/stage2_512.yaml")
    dtype = torch.float16
    scheduler = DDIMScheduler.from_pretrained(str(APP_ROOT / "configs"))
    audio_encoder = Audio2Feature(
        model_path=str(APP_ROOT / "checkpoints/whisper/tiny.pt"),
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )
    vae = AutoencoderKL.from_pretrained(
        str(APP_ROOT / "checkpoints/sd-vae-ft-mse"), torch_dtype=dtype
    )
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        str(APP_ROOT / "checkpoints/latentsync_unet.pt"),
        device="cpu",
    )
    unet = unet.to(dtype=dtype)
    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")
    if os.getenv("ENABLE_DEEPCACHE", "1") == "1":
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()
        pipeline._deepcache_helper = helper
    return config, pipeline, time.perf_counter() - started


CONFIG, PIPELINE, MODEL_LOAD_SECONDS = _load_pipeline()


def handler(job: dict) -> dict:
    payload = job.get("input") or {}
    job_id = _safe_id(payload.get("job_id", uuid.uuid4().hex), "job_id")
    avatar_id = _safe_id(payload.get("avatar_id", "default"), "avatar_id")
    video_url = payload.get("source_video_url")
    audio_url = payload.get("audio_url")
    if not video_url or not audio_url:
        raise ValueError("source_video_url and audio_url are required")

    max_duration = _bounded_int(payload, "max_duration_seconds", 60, 1, 600)
    inference_steps = _bounded_int(payload, "inference_steps", 20, 20, 50)
    guidance_scale = _bounded_float(payload, "guidance_scale", 1.5, 1.0, 3.0)
    seed = _bounded_int(payload, "seed", 1247, -1, 2_147_483_647)
    refresh_avatar = bool(payload.get("refresh_avatar", False))

    started = time.perf_counter()
    work = Path("/tmp") / f"latentsync-{job_id}-{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True, exist_ok=False)
    source_download = work / "source-download"
    audio_download = work / "audio-download"
    normalized_audio = work / "audio.wav"
    generated = work / "generated.mp4"
    final_video = work / "final.mp4"

    try:
        download_started = time.perf_counter()
        _, audio_bytes = _download(audio_url, audio_download)
        audio_duration = min(_probe_duration(audio_download), float(max_duration))
        if audio_duration <= 0:
            raise ValueError("audio duration must be greater than zero")
        _normalize_audio(audio_download, normalized_audio, audio_duration)

        cached_source = CACHE_ROOT / "sources" / f"{avatar_id}.mp4"
        source_cache_hit = cached_source.exists() and not refresh_avatar
        if not source_cache_hit:
            lock = FileLock(str(CACHE_ROOT / "sources" / f"{avatar_id}.lock"))
            with lock:
                if refresh_avatar or not cached_source.exists():
                    _, _ = _download(video_url, source_download)
                    temp_cached = cached_source.with_suffix(".tmp.mp4")
                    source_duration = min(_probe_duration(source_download), 600.0)
                    if source_duration <= 0:
                        raise ValueError("source video duration must be greater than zero")
                    _normalize_video(source_download, temp_cached, source_duration)
                    temp_cached.replace(cached_source)
                else:
                    source_cache_hit = True
        download_seconds = time.perf_counter() - download_started

        if seed == -1:
            generator = None
            effective_seed = int(torch.seed())
        else:
            effective_seed = seed
            generator = torch.Generator(device="cuda").manual_seed(seed)

        inference_started = time.perf_counter()
        with MODEL_LOCK:
            PIPELINE(
                video_path=str(cached_source),
                audio_path=str(normalized_audio),
                video_out_path=str(generated),
                num_frames=CONFIG.data.num_frames,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                weight_dtype=torch.float16,
                width=CONFIG.data.resolution,
                height=CONFIG.data.resolution,
                mask_image_path=CONFIG.data.mask_image_path,
                temp_dir=str(work / "pipeline-temp"),
                generator=generator,
            )
        inference_seconds = time.perf_counter() - inference_started

        _finalize_video(generated, final_video)
        object_key, output_url = _publish(final_video, job_id)
        return {
            "status": "completed",
            "job_id": job_id,
            "avatar_id": avatar_id,
            "video_url": output_url,
            "object_key": object_key,
            "audio_duration_seconds": round(audio_duration, 3),
            "audio_bytes": audio_bytes,
            "source_cache_hit": source_cache_hit,
            "model": "ByteDance/LatentSync-1.6",
            "generation_settings": {
                "inference_steps": inference_steps,
                "guidance_scale": guidance_scale,
                "seed": effective_seed,
                "deepcache": os.getenv("ENABLE_DEEPCACHE", "1") == "1",
            },
            "model_load_seconds": round(MODEL_LOAD_SECONDS, 3),
            "download_seconds": round(download_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "processing_time_seconds": round(time.perf_counter() - started, 3),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


runpod.serverless.start({"handler": handler})
