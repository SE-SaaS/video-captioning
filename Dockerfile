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

# Track 2 injects NO env vars, so the API keys must live inside the image.
# Pass them at build time:
#   --build-arg FIREWORKS_API_KEY=fw_xxx  --build-arg GEMINI_API_KEY=xxx
# The ensemble uses Fireworks (minimax, qwen) AND Google (gemini-3.5-flash), so BOTH are needed.
# NOTE: these keys are readable by anyone who pulls the public image (and via `docker history`).
# Rotate/delete them right after judging.
ARG FIREWORKS_API_KEY
ENV FIREWORKS_API_KEY=${FIREWORKS_API_KEY}
ARG GEMINI_API_KEY
ENV GEMINI_API_KEY=${GEMINI_API_KEY}

# Reads /input/tasks.json, writes /output/results.json.
ENTRYPOINT ["python", "-m", "Source.Main"]
