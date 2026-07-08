# FireworksClient: sends frames + prompts to the Fireworks chat API, with retries.
import base64
import os
import time

import requests


class FFireworksClient:
    def __init__(
        self,
        ModelId: str,
        BaseUrl: str,
        TimeoutSeconds: int,
        MaxRetries: int,
        BackoffSeconds: float,
        MaxTokens: int,
        Temperature: float,
        ReasoningEffort: str = "none",
        ApiKey: str | None = None,
    ) -> None:
        self.ModelId: str = ModelId
        self.BaseUrl: str = BaseUrl.rstrip("/")
        self.TimeoutSeconds: int = TimeoutSeconds
        self.MaxRetries: int = MaxRetries
        self.BackoffSeconds: float = BackoffSeconds
        self.MaxTokens: int = MaxTokens
        self.Temperature: float = Temperature
        self.ReasoningEffort: str = ReasoningEffort

        # Read the secret from the environment; allow an explicit override for tests.
        self.ApiKey: str = ApiKey or os.environ.get("FIREWORKS_API_KEY", "")
        if not self.ApiKey:
            raise RuntimeError("FIREWORKS_API_KEY is not set in the environment.")

    def BuildImageContent(self, FrameBytes: bytes) -> dict:
        EncodedFrame: str = base64.b64encode(FrameBytes).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{EncodedFrame}"},
        }

    def Complete(self, SystemPrompt: str, UserPrompt: str, Frames: list[bytes]) -> str:
        MessageContent: list[dict] = [{"type": "text", "text": UserPrompt}]
        for FrameData in Frames:
            MessageContent.append(self.BuildImageContent(FrameData))

        Messages: list[dict] = []
        if SystemPrompt:
            Messages.append({"role": "system", "content": SystemPrompt})
        Messages.append({"role": "user", "content": MessageContent})

        RequestPayload: dict = {
            "model": self.ModelId,
            "max_tokens": self.MaxTokens,
            "temperature": self.Temperature,
            "messages": Messages,
        }
        if self.ReasoningEffort:
            RequestPayload["reasoning_effort"] = self.ReasoningEffort
        return self.PostWithRetries(RequestPayload)

    def PostWithRetries(self, RequestPayload: dict) -> str:
        Endpoint: str = f"{self.BaseUrl}/chat/completions"
        RequestHeaders: dict = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.ApiKey}",
        }

        LastError: Exception | None = None
        for AttemptIndex in range(self.MaxRetries):
            try:
                Response = requests.post(
                    Endpoint,
                    headers=RequestHeaders,
                    json=RequestPayload,
                    timeout=self.TimeoutSeconds,
                )
                Response.raise_for_status()
                ResponseData: dict = Response.json()
                return ResponseData["choices"][0]["message"]["content"].strip()
            except (requests.RequestException, KeyError, IndexError) as CaughtError:
                LastError = CaughtError
                bHasMoreAttempts: bool = AttemptIndex < self.MaxRetries - 1
                if bHasMoreAttempts:
                    time.sleep(self.BackoffSeconds * (2**AttemptIndex))

        raise RuntimeError(
            f"Fireworks request failed after {self.MaxRetries} attempts: {LastError}"
        )
