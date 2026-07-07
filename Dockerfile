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

# Reads /input/tasks.json, writes /output/results.json; FIREWORKS_API_KEY injected at runtime.
ENTRYPOINT ["python", "-m", "Source.Main"]
