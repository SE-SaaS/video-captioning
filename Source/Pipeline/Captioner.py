# Captioner: builds the styled system prompt and asks the client for one caption.
from Source.Pipeline.FireworksClient import FFireworksClient
from Source.Pipeline.SystemPrompt import BuildSystemPrompt
from Source.Schema.Models import ECaptionStyle


class FCaptioner:
    def __init__(self, Client: FFireworksClient) -> None:
        self.Client: FFireworksClient = Client

    def GenerateCaption(self, Frames: list[bytes], Style: ECaptionStyle) -> str:
        SystemPrompt: str = BuildSystemPrompt(Style.value)
        UserPrompt: str = "Caption this video."
        return self.Client.Complete(SystemPrompt, UserPrompt, Frames)
