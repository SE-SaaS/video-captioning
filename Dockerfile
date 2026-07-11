FROM python:3.12-slim

# ffmpeg brings both ffmpeg and ffprobe, which FrameSampler shells out to.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml ./
COPY Source ./Source

# Track 2 injects NO env vars, so the Fireworks key must live inside the image.
# Pass it at build time:  --build-arg FIREWORKS_API_KEY=fw_xxx
# NOTE: this key is readable by anyone who pulls the public image (and via `docker history`).
# Rotate/delete it right after judging.
ARG FIREWORKS_API_KEY
ENV FIREWORKS_API_KEY=${FIREWORKS_API_KEY}

# Reads /input/tasks.json, writes /output/results.json.
ENTRYPOINT ["python", "-m", "Source.Main"]
