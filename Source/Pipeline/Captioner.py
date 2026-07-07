from Source.Pipeline.FireworksClient import FFireworksClient
from Source.Schema.Models import ECaptionStyle


# Placeholder prompts so the pipeline runs end-to-end today. Partner B replaces
# this map (and the selection logic) with the tuned style prompts + few-shot examples.
StubStyleInstructions: dict[ECaptionStyle, str] = {
    ECaptionStyle.Formal: "Write a professional, objective, factual caption.",
    ECaptionStyle.Sarcastic: "Write a dry, ironic, lightly mocking caption.",
    ECaptionStyle.HumorousTech: "Write a funny caption with a technology or programming reference.",
    ECaptionStyle.HumorousNonTech: "Write a funny, everyday caption with no technical jargon.",
}

StubSystemPrompt: str = (
    "You caption short video clips from a handful of sampled frames. "
    "Describe what actually happens in the clip in a single sentence."
)


class FCaptioner:
    def __init__(self, Client: FFireworksClient) -> None:
        self.Client: FFireworksClient = Client

    def BuildUserPrompt(self, Style: ECaptionStyle) -> str:
        return StubStyleInstructions[Style]

    def GenerateCaption(self, Frames: list[bytes], Style: ECaptionStyle) -> str:
        UserPrompt: str = self.BuildUserPrompt(Style)
        return self.Client.Complete(StubSystemPrompt, UserPrompt, Frames)
