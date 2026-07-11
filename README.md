# Multi-Style Video Captioner

A small agent that watches a short video clip and writes a caption in a requested
style. Built for the AMD Developer Hackathon (Track 2).

## Styles

- `formal` — professional and factual
- `sarcastic` — dry and ironic
- `humorous_tech` — jokes with a tech/programming flavor
- `humorous_non_tech` — everyday humor

## How it works

1. Sample frames from the clip with ffmpeg (spaced across the video, downscaled).
2. Send the frames to a VLM (vision language model, eg. Qwen3-Plus via Fireworks AI)
   with a per-style prompt.
3. Write one caption per requested style.

Everything is set in [`config.yaml`](config.yaml): the model, frame sampling,
and per-style temperatures.

### Ensembling (optional)

Instead of one VLM, you can list several. Each VLM captions every style, then a
judge model picks the best caption or merges their best parts into the final one.
List one model for single-model mode, or two or more to turn on the ensemble.

## Input / output

Reads tasks from `/input/tasks.json`:

```json
[
  { "task_id": "v1", "video_url": "https://.../clip.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"] }
]
```

Writes results to `/output/results.json`:

```json
[
  { "task_id": "v1", "captions": { "formal": "...", "sarcastic": "..." } }
]
```

## Run locally

```bash
pip install -r requirements.txt
# put FIREWORKS_API_KEY in a .env file
python -m Source.Main
```

Paths are picked automatically: `/input` and `/output` inside Docker,
`io/input` and `io/output` locally.

## Run with Docker

```bash
docker buildx build --platform linux/amd64 \
  --build-arg FIREWORKS_API_KEY=your_key \
  --tag video-captioning:latest --load .

docker run --rm -v "$PWD/io/input:/input" -v "$PWD/io/output:/output" \
  video-captioning:latest
```

## Testing

A local harness captions a set of sample clips and scores each caption on
accuracy and style match:

```bash
python Testing/run_tests.py
```
