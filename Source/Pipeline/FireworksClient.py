# FireworksClient: sends frames + prompts to the Fireworks chat API, with retries.
import base64
import os
import re
import threading
import time

import requests

# Some models leak chain-of-thought inline as <think>...</think> in the content field.
# Strip it so only the final answer remains (reasoning stays enabled server-side).
ThinkBlock = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# ---- Global request throttle -------------------------------------------------
# Fireworks caps accounts with NO payment method at 10 requests/minute account-wide.
# This spaces out EVERY API call across all clients/threads so their combined rate
# stays under the limit. Disabled (0) by default; the local test harness turns it on.
_ThrottleLock = threading.Lock()
_MinRequestInterval: float = 0.0
_NextAllowedTime: float = 0.0


def SetMinRequestInterval(Seconds: float) -> None:
    """Set the minimum seconds between any two API calls (0 disables throttling)."""
    global _MinRequestInterval
    _MinRequestInterval = max(0.0, float(Seconds))


def AwaitRequestSlot() -> None:
    # Reserve the next evenly-spaced slot under the lock, then sleep to it outside the
    # lock, so concurrent threads queue up instead of all firing at once.
    if _MinRequestInterval <= 0.0:
        return
    global _NextAllowedTime
    with _ThrottleLock:
        Now: float = time.monotonic()
        SlotTime: float = max(Now, _NextAllowedTime)
        _NextAllowedTime = SlotTime + _MinRequestInterval
    WaitFor: float = SlotTime - time.monotonic()
    if WaitFor > 0:
        time.sleep(WaitFor)


def StripReasoning(Text: str) -> str:
    Cleaned = ThinkBlock.sub("", Text)
    # If thinking was truncated so only a closing tag survives, keep what follows it.
    if "</think>" in Cleaned.lower():
        Index = Cleaned.lower().rindex("</think>") + len("</think>")
        Cleaned = Cleaned[Index:]
    # If only an opening tag survives (no close), drop everything from it onward.
    Lower = Cleaned.lower()
    if "<think>" in Lower:
        Cleaned = Cleaned[: Lower.index("<think>")]
    return Cleaned.strip()


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
        ApiKeyEnv: str = "FIREWORKS_API_KEY",
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

        # Read the secret from the configured env var; allow an explicit override for tests.
        self.ApiKey: str = ApiKey or os.environ.get(ApiKeyEnv, "")
        if not self.ApiKey:
            raise RuntimeError(f"{ApiKeyEnv} is not set in the environment.")

    def BuildImageContent(self, FrameBytes: bytes) -> dict:
        EncodedFrame: str = base64.b64encode(FrameBytes).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{EncodedFrame}"},
        }

    def Complete(
        self,
        SystemPrompt: str,
        UserPrompt: str,
        Frames: list[bytes],
        Temperature: float | None = None,
    ) -> str:
        MessageContent: list[dict] = [{"type": "text", "text": UserPrompt}]
        for FrameData in Frames:
            MessageContent.append(self.BuildImageContent(FrameData))

        Messages: list[dict] = []
        if SystemPrompt:
            Messages.append({"role": "system", "content": SystemPrompt})
        Messages.append({"role": "user", "content": MessageContent})

        # Per-call temperature overrides the client default when provided.
        EffectiveTemperature: float = (
            Temperature if Temperature is not None else self.Temperature
        )
        RequestPayload: dict = {
            "model": self.ModelId,
            "max_tokens": self.MaxTokens,
            "temperature": EffectiveTemperature,
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
                AwaitRequestSlot()  # global rate-limit spacing (no-op when disabled)
                Response = requests.post(
                    Endpoint,
                    headers=RequestHeaders,
                    json=RequestPayload,
                    timeout=self.TimeoutSeconds,
                )
                Response.raise_for_status()
                ResponseData: dict = Response.json()
                Content: str = ResponseData["choices"][0]["message"]["content"]
                return StripReasoning(Content)
            except (requests.RequestException, KeyError, IndexError) as CaughtError:
                LastError = CaughtError
                bHasMoreAttempts: bool = AttemptIndex < self.MaxRetries - 1
                if bHasMoreAttempts:
                    time.sleep(self.RetryDelay(AttemptIndex, CaughtError))

        raise RuntimeError(
            f"Fireworks request failed after {self.MaxRetries} attempts: {LastError}"
        )

    def RetryDelay(self, AttemptIndex: int, Error: Exception) -> float:
        # On a 429 (rate limit), honor the server's Retry-After header if present;
        # otherwise back off harder than normal so we drop below the limit.
        Response = getattr(Error, "response", None)
        ExpBackoff: float = self.BackoffSeconds * (2**AttemptIndex)
        if Response is not None and Response.status_code == 429:
            RetryAfter: str = Response.headers.get("Retry-After", "").strip()
            if RetryAfter.isdigit():
                return float(RetryAfter)
            return max(ExpBackoff, _MinRequestInterval, 6.5)
        return ExpBackoff
