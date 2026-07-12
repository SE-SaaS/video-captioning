# Captioner: generates a caption per style. Single-model mode uses the one model
# directly; ensemble mode (2+ models) collects each model's candidate caption and
# hands them to a judge model that returns the final caption.
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from Source.Pipeline.FireworksClient import FFireworksClient
from Source.Pipeline.SystemPrompt import BuildJudgeSystemPrompt, BuildSystemPrompt
from Source.Schema.Models import ECaptionStyle


@dataclass
class FCaptionTrace:
    Style: str
    Candidates: list[tuple[str, str]]  # (model_id, candidate_caption) per ensemble member
    FinalCaption: str
    JudgeModelId: str | None = None    # None in single-model mode
    DurationSeconds: float = 0.0       # wall-time to produce this caption (candidates + judge)
    CandidateDurations: list[float] = field(default_factory=list)  # per-model call time, aligned to Candidates
    JudgeDurationSeconds: float = 0.0  # judge call time (0 in single-model mode)


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

    def GenerateCaption(self, Frames: list[bytes], Style: ECaptionStyle) -> FCaptionTrace:
        Start: float = time.monotonic()
        SystemPrompt: str = BuildSystemPrompt(Style.value)
        UserPrompt: str = "Caption this video."
        # Fall back to each client's default temperature if the style is unlisted.
        Temperature: float | None = self.StyleTemperatures.get(Style.value)

        def RunClient(Client: FFireworksClient) -> tuple[str, str, float]:
            # Time each model's own call so the report can show per-model latency.
            CallStart: float = time.monotonic()
            Caption: str = Client.Complete(SystemPrompt, UserPrompt, Frames, Temperature)
            return (Client.ModelId, Caption, time.monotonic() - CallStart)

        def RunChain(Clients: list[FFireworksClient]) -> list[tuple[str, str, float]]:
            return [RunClient(Client) for Client in Clients]

        # Two-lane execution: the first (slowest) model runs alone in Lane A, while the
        # remaining models run one-after-another in Lane B — the two lanes in parallel. So a
        # caption takes max(lane A, lane B) + judge, but only 2 extra threads exist at a time
        # (safe under a tight PID/thread limit). Order is preserved: A's model, then B's.
        if len(self.Clients) <= 1:
            Results: list[tuple[str, str, float]] = RunChain(self.Clients)
        else:
            with ThreadPoolExecutor(max_workers=2) as Executor:
                LaneA = Executor.submit(RunChain, self.Clients[:1])
                LaneB = Executor.submit(RunChain, self.Clients[1:])
                Results = LaneA.result() + LaneB.result()

        Candidates: list[tuple[str, str]] = [(Mid, Cap) for Mid, Cap, _ in Results]
        CandidateDurations: list[float] = [Dur for _, _, Dur in Results]

        # Single-model mode: no judge, the only caption is final.
        if len(Candidates) == 1:
            Trace = FCaptionTrace(Style.value, Candidates, Candidates[0][1])
        else:
            JudgeStart: float = time.monotonic()
            FinalCaption: str = self.JudgeCaptions(
                Frames, Style, [Caption for _, Caption in Candidates]
            )
            Trace = FCaptionTrace(
                Style.value, Candidates, FinalCaption, self.JudgeClient.ModelId
            )
            Trace.JudgeDurationSeconds = time.monotonic() - JudgeStart
        Trace.CandidateDurations = CandidateDurations
        Trace.DurationSeconds = time.monotonic() - Start
        return Trace

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
