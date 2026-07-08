# Dev utility: probe the clips in ./videos/, print their durations, and save a
# short report (per-clip duration + mean and max) to scripts/video_stats.txt.
import glob
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "test_videos_stats.txt")
VIDEOS_DIR = "videos"


def probe_duration_seconds(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def main() -> None:
    paths = sorted(glob.glob(os.path.join(VIDEOS_DIR, "*.mp4")))
    lines = []
    durations = []
    for path in paths:
        seconds = probe_duration_seconds(path)
        durations.append(seconds)
        lines.append(f"{os.path.basename(path):20s} {seconds:8.2f} s")

    if durations:
        mean = sum(durations) / len(durations)
        maximum = max(durations)
        lines.append("-" * 32)
        lines.append(f"{'count':20s} {len(durations):8d}")
        lines.append(f"{'mean duration':20s} {mean:8.2f} s")
        lines.append(f"{'max duration':20s} {maximum:8.2f} s")
    else:
        lines.append("No .mp4 files found in ./videos/")

    report = "\n".join(lines)
    print(report)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
