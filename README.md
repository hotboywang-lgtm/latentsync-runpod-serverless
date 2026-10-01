# LatentSync 1.6 RunPod Serverless worker

Drop-in video-generation replacement for the existing MuseTalk endpoint. It
accepts a source-video URL and an already-generated audio URL, then uploads the
result to S3-compatible object storage. It does not perform script generation
or TTS.

## Request

```json
{
  "input": {
    "job_id": "video-001",
    "avatar_id": "presenter-001",
    "source_video_url": "https://example.com/source.mp4",
    "audio_url": "https://example.com/audio.wav",
    "max_duration_seconds": 60,
    "inference_steps": 20,
    "guidance_scale": 1.5,
    "seed": 1247
  }
}
```

`inference_steps` accepts 10–50. The official-quality default is 20; lower
values are useful for speed/quality benchmarking.

The response preserves the existing `video_url` output contract. Source video
downloads are cached by `avatar_id` on the RunPod Network Volume. Set
`refresh_avatar` to `true` when replacing an avatar's source material.

## Runtime environment

- `OUTPUT_MODE=s3`
- `CACHE_ROOT=/runpod-volume/latentsync-cache`
- `S3_BUCKET`
- `S3_ENDPOINT_URL`
- `S3_REGION`
- `S3_PREFIX=latentsync`
- `S3_SIGNATURE_VERSION=s3` for Alibaba OSS Signature V2
- `S3_ACCESS_KEY_ID` and `S3_SECRET_ACCESS_KEY` via RunPod Secrets
- `PRESIGNED_URL_TTL=86400`
- `ENABLE_DEEPCACHE=1`
