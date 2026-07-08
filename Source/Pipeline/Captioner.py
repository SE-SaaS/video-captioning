# Captioner: builds the styled system prompt and asks the client for one caption.
from Source.Pipeline.FireworksClient import FFireworksClient
from Source.Pipeline.SystemPrompt import BuildSystemPrompt
from Source.Schema.Models import ECaptionStyle


class FCaptioner:
    def __init__(
        self,
        Client: FFireworksClient,
        StyleTemperatures: dict[str, float],
    ) -> None:
        self.Client: FFireworksClient = Client
        self.StyleTemperatures: dict[str, float] = StyleTemperatures

    def GenerateCaption(self, Frames: list[bytes], Style: ECaptionStyle) -> str:
        SystemPrompt: str = BuildSystemPrompt(Style.value)
        UserPrompt: str = "Caption this video."
        # Fall back to the client default temperature if the style is unlisted.
        Temperature: float | None = self.StyleTemperatures.get(Style.value)
        return self.Client.Complete(SystemPrompt, UserPrompt, Frames, Temperature)
