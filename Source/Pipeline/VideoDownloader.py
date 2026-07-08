# VideoDownloader: fetches a remote video to the work dir (fallback for the stream path).
import os
import time

import requests


class FVideoDownloader:
    def __init__(
        self,
        WorkDir: str,
        TimeoutSeconds: int,
        MaxRetries: int,
        BackoffSeconds: float,
    ) -> None:
        self.WorkDir: str = WorkDir
        self.TimeoutSeconds: int = TimeoutSeconds
        self.MaxRetries: int = MaxRetries
        self.BackoffSeconds: float = BackoffSeconds
        os.makedirs(self.WorkDir, exist_ok=True)

    def DownloadVideo(self, VideoUrl: str, TaskId: str) -> str:
        DestinationPath: str = os.path.join(self.WorkDir, f"{TaskId}.mp4")

        LastError: Exception | None = None
        for AttemptIndex in range(self.MaxRetries):
            try:
                with requests.get(
                    VideoUrl, stream=True, timeout=self.TimeoutSeconds
                ) as Response:
                    Response.raise_for_status()
                    with open(DestinationPath, "wb") as VideoFile:
                        for Chunk in Response.iter_content(chunk_size=1 << 16):
                            if Chunk:
                                VideoFile.write(Chunk)
                return DestinationPath
            except requests.RequestException as CaughtError:
                LastError = CaughtError
                bHasMoreAttempts: bool = AttemptIndex < self.MaxRetries - 1
                if bHasMoreAttempts:
                    time.sleep(self.BackoffSeconds * (2**AttemptIndex))

        raise RuntimeError(
            f"Failed to download '{VideoUrl}' after {self.MaxRetries} attempts: {LastError}"
        )
