# Captioner: generates a caption per style. Single-model mode uses the one model
# directly; ensemble mode (2+ models) collects each model's candidate caption and
# hands them to a judge model that returns the final caption.
from Source.Pipeline.FireworksClient import FFireworksClient
from Source.Pipeline.SystemPrompt import BuildJudgeSystemPrompt, BuildSystemPrompt
from Source.Schema.Models import ECaptionStyle


class FCaptioner:
    def __init__(
        self,
        Clients: list[FFireworksClient],
        JudgeClient: FFireworksClient,
        JudgePassFrames: bool,
        StyleTemperatures: dict[str, float],
    ) -> None:
        self.Clients: list[FFireworksClient] = Clients
        self.JudgeClient: FFireworksClient = JudgeClient
        self.JudgePassFrames: bool = JudgePassFrames
        self.StyleTemperatures: dict[str, float] = StyleTemperatures

    def GenerateCaption(self, Frames: list[bytes], Style: ECaptionStyle) -> str:
        SystemPrompt: str = BuildSystemPrompt(Style.value)
        UserPrompt: str = "Caption this video."
        # Fall back to each client's default temperature if the style is unlisted.
        Temperature: float | None = self.StyleTemperatures.get(Style.value)

        Candidates: list[str] = [
            Client.Complete(SystemPrompt, UserPrompt, Frames, Temperature)
            for Client in self.Clients
        ]

        # Single-model mode: no judge, return the only caption.
        if len(Candidates) == 1:
            return Candidates[0]

        return self.JudgeCaptions(Frames, Style, Candidates)

    def JudgeCaptions(
        self,
        Frames: list[bytes],
        Style: ECaptionStyle,
        Candidates: list[str],
    ) -> str:
        JudgeSystemPrompt: str = BuildJudgeSystemPrompt(Style.value)
        NumberedCandidates: str = "\n".join(
            f"Candidate {Index + 1}: {Candidate}"
            for Index, Candidate in enumerate(Candidates)
        )
        JudgeUserPrompt: str = (
            f"Style: {Style.value}\n"
            f"{NumberedCandidates}\n\n"
            "Return the single best final caption."
        )
        JudgeFrames: list[bytes] = Frames if self.JudgePassFrames else []
        return self.JudgeClient.Complete(JudgeSystemPrompt, JudgeUserPrompt, JudgeFrames)
