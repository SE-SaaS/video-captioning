# FrameSampler: uses ffmpeg to sample and downscale frames from a video (URL or path).
import math
import subprocess


class FFrameSampler:
    # Fireworks accepts at most 30 images per request; never emit more than this.
    MaxImagesPerRequest: int = 30

    def __init__(
        self,
        PerThirtySeconds: int,
        MaxTotal: int,
        Width: int,
        JpegQuality: int,
        MaxPayloadMB: float,
        TimeoutSeconds: int,
    ) -> None:
        self.PerThirtySeconds: int = PerThirtySeconds
        self.MaxTotal: int = MaxTotal
        self.Width: int = Width
        self.JpegQuality: int = JpegQuality
        self.MaxPayloadMB: float = MaxPayloadMB
        self.TimeoutSeconds: int = TimeoutSeconds

    def ProbeDurationSeconds(self, InputSource: str) -> float:
        ProbeArgs: list[str] = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            InputSource,
        ]
        ProbeResult = subprocess.run(
            ProbeArgs,
            capture_output=True,
            text=True,
            timeout=self.TimeoutSeconds,
        )
        if ProbeResult.returncode != 0 or not ProbeResult.stdout.strip():
            raise RuntimeError(f"ffprobe failed for '{InputSource}': {ProbeResult.stderr.strip()}")
        return float(ProbeResult.stdout.strip())

    def ComputeFrameCount(self, DurationSeconds: float) -> int:
        RawCount: int = math.ceil(self.PerThirtySeconds * DurationSeconds / 30.0)
        Ceiling: int = min(self.MaxTotal, self.MaxImagesPerRequest)
        return max(1, min(RawCount, Ceiling))

    def ComputeTimestamps(self, DurationSeconds: float, FrameCount: int) -> list[float]:
        # Sample at segment centers so we never grab a black first/last frame.
        return [DurationSeconds * (Index + 0.5) / FrameCount for Index in range(FrameCount)]

    def ExtractFrame(self, InputSource: str, TimestampSeconds: float) -> bytes:
        # 0-100 quality maps onto ffmpeg mjpeg's inverted 2-31 qscale (2 = best).
        QScale: int = max(2, min(31, round(31 - (self.JpegQuality / 100.0) * 29)))
        ExtractArgs: list[str] = [
            "ffmpeg",
            "-nostdin",
            "-ss",
            f"{TimestampSeconds:.3f}",
            "-i",
            InputSource,
            "-frames:v",
            "1",
            "-vf",
            f"scale=w='min({self.Width},iw)':h=-2",
            "-q:v",
            str(QScale),
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]
        ExtractResult = subprocess.run(
            ExtractArgs,
            capture_output=True,
            timeout=self.TimeoutSeconds,
        )
        if ExtractResult.returncode != 0 or not ExtractResult.stdout:
            raise RuntimeError(
                f"ffmpeg frame extraction failed at {TimestampSeconds:.3f}s "
                f"for '{InputSource}': {ExtractResult.stderr.decode('utf-8', 'ignore').strip()}"
            )
        return ExtractResult.stdout

    def EnforcePayloadLimit(self, Frames: list[bytes]) -> list[bytes]:
        # base64 inflates raw bytes by ~4/3, so budget the raw size accordingly.
        BudgetBytes: int = int(self.MaxPayloadMB * 1_000_000 * 3 / 4)
        TotalBytes: int = sum(len(Frame) for Frame in Frames)
        if TotalBytes <= BudgetBytes or len(Frames) <= 1:
            return Frames

        AverageSize: float = TotalBytes / len(Frames)
        KeepCount: int = max(1, int(BudgetBytes / AverageSize))
        if KeepCount >= len(Frames):
            return Frames

        KeptIndices: list[int] = [
            round(Index * (len(Frames) - 1) / (KeepCount - 1)) if KeepCount > 1 else 0
            for Index in range(KeepCount)
        ]
        return [Frames[Index] for Index in sorted(set(KeptIndices))]

    def SampleFrames(self, InputSource: str) -> list[bytes]:
        DurationSeconds: float = self.ProbeDurationSeconds(InputSource)
        FrameCount: int = self.ComputeFrameCount(DurationSeconds)
        Timestamps: list[float] = self.ComputeTimestamps(DurationSeconds, FrameCount)
        Frames: list[bytes] = [
            self.ExtractFrame(InputSource, Timestamp) for Timestamp in Timestamps
        ]
        return self.EnforcePayloadLimit(Frames)
