# Local test harness: captions all clips in Testing/tasks.json using the CURRENT root
# config.yaml pipeline (ensemble + judge), then a separate scorer LLM (Testing/test_config.yaml)
# rates each final caption on accuracy + style_match (0-1), mirroring the hackathon rubric.
#
# Run from anywhere:  python Testing/run_tests.py
# Captioning uses the root config.yaml; only the scorer + test I/O come from test_config.yaml.
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from types import SimpleNamespace

import yaml

# Make the repo root importable and the CWD, so Source.* imports and the relative
# TestVideos/ paths both resolve the same way regardless of where this is launched.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from Source.Config import LoadConfig  # noqa: E402
from Source.Main import (  # noqa: E402
    AcquireAllFrames,
    BuildClient,
    FormatDuration,
    GenerateAllCaptions,
    LoadDotEnvIfPresent,
)
from Source.Pipeline.FireworksClient import (  # noqa: E402
    EnableLatencyTracking,
    GetLatencyRecords,
    GetRetryRecords,
    SetMinRequestInterval,
)
from Source.Pipeline.FrameSampler import FFrameSampler  # noqa: E402
from Source.Pipeline.SystemPrompt import BuildScorerSystemPrompt  # noqa: E402
from Source.Pipeline.VideoDownloader import FVideoDownloader  # noqa: E402
from Source.Schema.IOSchema import LoadTasks  # noqa: E402
from Source.Schema.Models import ECaptionStyle  # noqa: E402

TEST_CONFIG_PATH = "Testing/test_config.yaml"
ReportWidth = 92


class LiveProgress:
    """One in-place updating line per phase (uses \\r), so long runs stay readable."""

    BarWidth = 24

    def __init__(self, Label: str) -> None:
        self.Label = Label
        self.Width = 0
        print(f"\n  {Label}")

    def Update(self, Done: int, Total: int, Item: str) -> None:
        Frac = Done / Total if Total else 1.0
        Filled = int(Frac * self.BarWidth)
        Bar = "[" + "#" * Filled + "-" * (self.BarWidth - Filled) + "]"
        Line = f"    {Bar} {Done:>3}/{Total} ({Frac * 100:5.1f}%)  | {Item}"
        # Pad to erase any leftover from a previous longer line.
        Pad = max(0, self.Width - len(Line))
        sys.stdout.write("\r" + Line + " " * Pad)
        sys.stdout.flush()
        self.Width = len(Line)
        if Done >= Total:
            sys.stdout.write("\n")
            sys.stdout.flush()


def LoadTestConfig() -> dict:
    with open(TEST_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ParseScores(Text: str) -> tuple[float, float]:
    """Pull {"accuracy":..,"style_match":..} out of the scorer's reply, robustly."""
    Candidates = re.findall(r"\{[^{}]*\}", Text, re.DOTALL)
    for Chunk in reversed(Candidates):  # last JSON object is the answer if any preamble leaks
        try:
            Data = json.loads(Chunk)
            Acc = float(Data["accuracy"])
            Sm = float(Data["style_match"])
            return max(0.0, min(1.0, Acc)), max(0.0, min(1.0, Sm))
        except (ValueError, KeyError, TypeError):
            continue
    raise ValueError(f"Could not parse scores from: {Text[:200]!r}")


def ScoreCaption(ScorerClient, Frames, Caption, StyleValue, PassFrames) -> tuple[float, float]:
    SystemPrompt = BuildScorerSystemPrompt(StyleValue)
    UserPrompt = f'Requested style: {StyleValue}\nCaption to score: "{Caption}"\nReturn the JSON scores.'
    FramesToSend = Frames if PassFrames else []
    Reply = ScorerClient.Complete(SystemPrompt, UserPrompt, FramesToSend)
    return ParseScores(Reply)


def Mean(Values: list[float]) -> float:
    return sum(Values) / len(Values) if Values else 0.0


def main() -> int:
    StartTime = time.monotonic()
    LoadDotEnvIfPresent()
    RootConfig = LoadConfig()  # captioning pipeline settings (root config.yaml)
    TestConfig = LoadTestConfig()
    # Throttle every API call (captioning + judge + scorer) to stay under Fireworks' 10 RPM
    # cap on keys with no payment method. Configured in Testing/test_config.yaml.
    SetMinRequestInterval(float(TestConfig.get("MinRequestIntervalSeconds", 0.0)))
    ScorerCfg = TestConfig["Scorer"]
    Weights = TestConfig["Weights"]
    Paths = TestConfig["Paths"]
    WAcc = float(Weights["Accuracy"])
    WStyle = float(Weights["StyleMatch"])

    Tasks = LoadTasks(Paths["Tasks"], RootConfig.DefaultStyles)

    # Build the same pipeline Main uses, from the root config.
    Sampler = FFrameSampler(
        PerThirtySeconds=RootConfig.Frames.PerThirtySeconds,
        MaxTotal=RootConfig.Frames.MaxTotal,
        Width=RootConfig.Frames.Width,
        JpegQuality=RootConfig.Frames.JpegQuality,
        MaxPayloadMB=RootConfig.Frames.MaxPayloadMB,
        TimeoutSeconds=RootConfig.Client.TimeoutSeconds,
    )
    Downloader = FVideoDownloader(
        WorkDir=RootConfig.Paths.WorkDir,
        TimeoutSeconds=RootConfig.Client.TimeoutSeconds,
        MaxRetries=RootConfig.Client.MaxRetries,
        BackoffSeconds=RootConfig.Client.BackoffSeconds,
    )
    from Source.Pipeline.Captioner import FCaptioner

    Clients = [BuildClient(m, RootConfig.Client) for m in RootConfig.Models]
    JudgeClient = BuildClient(RootConfig.Judge, RootConfig.Client)
    Captioner = FCaptioner(
        Clients, JudgeClient, RootConfig.Judge.PassFrames, RootConfig.StyleTemperatures
    )

    TotalCaptions = sum(len(t.Styles) for t in Tasks)
    Mode = "ensemble" if len(RootConfig.Models) > 1 else "single-model"
    print("=" * ReportWidth)
    print(f" RUNNING TEST  |  {len(Tasks)} clips  |  {TotalCaptions} captions  |  pipeline: {Mode}")
    print(f" Models: {', '.join(m.Id.rsplit('/', 1)[-1] for m in RootConfig.Models)}"
          + (f"  ->  judge: {RootConfig.Judge.Id.rsplit('/', 1)[-1]}" if len(RootConfig.Models) > 1 else ""))
    print(f" Scorer: {ScorerCfg['Id'].rsplit('/', 1)[-1]}")
    print("=" * ReportWidth)

    FrameProg = LiveProgress("[1/3] Acquiring frames")
    FramesByTask = AcquireAllFrames(Tasks, RootConfig, Sampler, Downloader, FrameProg.Update)

    # Track per-request latency across the captioning phase (candidate + judge calls) so the
    # report can show avg/min/max and flag anything approaching the 30s per-request cap.
    EnableLatencyTracking()
    CapProg = LiveProgress("[2/3] Captioning (ensemble + judge)")
    TracesByTask = GenerateAllCaptions(Tasks, FramesByTask, RootConfig, Captioner, CapProg.Update)
    CaptionLatencies = GetLatencyRecords()  # per API call (candidate/judge)
    CaptionRetries = GetRetryRecords()      # (model, attempts, succeeded) for requests that retried
    CaptionDurations = [  # per FINAL caption (candidates + judge combined)
        Trace.DurationSeconds
        for StyleTraces in TracesByTask.values()
        for Trace in StyleTraces.values()
    ]

    # Scorer client (its own model / params from test_config.yaml).
    ScorerClient = BuildClient(SimpleNamespace(**ScorerCfg), RootConfig.Client)
    PassFrames = bool(ScorerCfg.get("PassFrames", True))

    ScoreTotal = sum(
        1 for Task in Tasks for Style in Task.Styles
        if TracesByTask.get(Task.TaskId, {}).get(Style.value) is not None
    )
    ScoreProg = LiveProgress("[3/3] Scoring final captions")
    ScoreDone = 0
    Records: list[dict] = []
    with ThreadPoolExecutor(max_workers=RootConfig.Runtime.MaxWorkers) as Executor:
        FutureMap = {}
        for Task in Tasks:
            Frames = FramesByTask.get(Task.TaskId) or []
            StyleTraces = TracesByTask.get(Task.TaskId, {})
            for Style in Task.Styles:
                Trace = StyleTraces.get(Style.value)
                if Trace is None:
                    continue  # captioning failed for this style; skip scoring
                Fut = Executor.submit(
                    ScoreCaption, ScorerClient, Frames, Trace.FinalCaption, Style.value, PassFrames
                )
                FutureMap[Fut] = (Task, Style.value, Trace)

        for Fut in as_completed(FutureMap):
            Task, StyleValue, Trace = FutureMap[Fut]
            try:
                Acc, Sm = Fut.result()
            except Exception as Err:
                print(f"[warn] scoring failed {Task.TaskId}/{StyleValue}: {Err}", file=sys.stderr)
                Acc, Sm = 0.0, 0.0
            ScoreDone += 1
            ScoreProg.Update(ScoreDone, ScoreTotal, f"{Task.TaskId}/{StyleValue}")
            Records.append({
                "task_id": Task.TaskId,
                "video_url": Task.VideoUrl,
                "style": StyleValue,
                "caption": Trace.FinalCaption,
                "candidates": [{"model": mid.rsplit("/", 1)[-1], "caption": c} for mid, c in Trace.Candidates],
                "judge_model": (Trace.JudgeModelId or "").rsplit("/", 1)[-1] or None,
                "accuracy": round(Acc, 3),
                "style_match": round(Sm, 3),
                "score": round(WAcc * Acc + WStyle * Sm, 3),
            })

    WriteReports(RootConfig, TestConfig, Records, time.monotonic() - StartTime,
                 CaptionLatencies, CaptionDurations, CaptionRetries)
    return 0


def LatencyStats(Records: "list[tuple[str, float]]") -> dict:
    """avg/min/max/count over all captioning requests, plus a per-model breakdown."""
    Secs = [s for _, s in Records]
    Overall = {
        "count": len(Secs),
        "avg": Mean(Secs),
        "min": min(Secs) if Secs else 0.0,
        "max": max(Secs) if Secs else 0.0,
    }
    PerModel: dict[str, dict] = {}
    for ModelId in sorted({m for m, _ in Records}):
        MSecs = [s for m, s in Records if m == ModelId]
        PerModel[ModelId.rsplit("/", 1)[-1]] = {
            "count": len(MSecs), "avg": Mean(MSecs), "min": min(MSecs), "max": max(MSecs),
        }
    return {"overall": Overall, "per_model": PerModel}


def WriteReports(RootConfig, TestConfig, Records: list[dict], ElapsedSeconds: float,
                 CaptionLatencies: "list[tuple[str, float]]",
                 CaptionDurations: "list[float]",
                 CaptionRetries: "list[tuple[str, int, bool]]") -> None:
    Paths = TestConfig["Paths"]
    Timestamp = datetime.now().isoformat(timespec="seconds")
    Latency = LatencyStats(CaptionLatencies)
    CaptionLatency = {
        "count": len(CaptionDurations),
        "avg": Mean(CaptionDurations),
        "min": min(CaptionDurations) if CaptionDurations else 0.0,
        "max": max(CaptionDurations) if CaptionDurations else 0.0,
    }
    RetryInfo = {
        "retried_requests": sum(1 for _, _, ok in CaptionRetries if ok),
        "failed_requests": sum(1 for _, _, ok in CaptionRetries if not ok),
        "max_attempts": max((a for _, a, _ in CaptionRetries), default=1),
        "by_model": {
            ModelId.rsplit("/", 1)[-1]: {
                "retried": sum(1 for m, _, ok in CaptionRetries if m == ModelId and ok),
                "failed": sum(1 for m, _, ok in CaptionRetries if m == ModelId and not ok),
            }
            for ModelId in sorted({m for m, _, _ in CaptionRetries})
        },
    }

    # Aggregates.
    Overall = {
        "accuracy": Mean([r["accuracy"] for r in Records]),
        "style_match": Mean([r["style_match"] for r in Records]),
        "score": Mean([r["score"] for r in Records]),
    }
    Styles = [s.value for s in ECaptionStyle]
    PerStyle = {
        s: {
            "accuracy": Mean([r["accuracy"] for r in Records if r["style"] == s]),
            "style_match": Mean([r["style_match"] for r in Records if r["style"] == s]),
            "score": Mean([r["score"] for r in Records if r["style"] == s]),
            "n": sum(1 for r in Records if r["style"] == s),
        }
        for s in Styles
    }
    Clips = sorted({r["task_id"] for r in Records})
    PerClip = {
        c: Mean([r["score"] for r in Records if r["task_id"] == c]) for c in Clips
    }

    # Config snapshot so each run is self-describing.
    ConfigSnapshot = {
        "mode": "ensemble" if len(RootConfig.Models) > 1 else "single-model",
        "models": [
            {"provider": m.Provider, "id": m.Id, "temperature": m.Temperature,
             "max_tokens": m.MaxTokens, "reasoning_effort": m.ReasoningEffort}
            for m in RootConfig.Models
        ],
        "judge": (
            {"provider": RootConfig.Judge.Provider, "id": RootConfig.Judge.Id,
             "temperature": RootConfig.Judge.Temperature, "max_tokens": RootConfig.Judge.MaxTokens,
             "reasoning_effort": RootConfig.Judge.ReasoningEffort,
             "pass_frames": RootConfig.Judge.PassFrames}
            if len(RootConfig.Models) > 1 else None
        ),
        "style_temperatures": RootConfig.StyleTemperatures,
        "frames": {
            "per_thirty_seconds": RootConfig.Frames.PerThirtySeconds,
            "max_total": RootConfig.Frames.MaxTotal, "width": RootConfig.Frames.Width,
            "jpeg_quality": RootConfig.Frames.JpegQuality, "source": RootConfig.Frames.Source,
        },
        "scorer": {**TestConfig["Scorer"]},
        "weights": {**TestConfig["Weights"]},
    }

    # Full detail -> JSON (overwritten each run).
    with open(Paths["ReportJson"], "w", encoding="utf-8") as f:
        json.dump(
            {"timestamp": Timestamp, "elapsed_seconds": round(ElapsedSeconds, 1),
             "config": ConfigSnapshot, "overall": Overall, "request_latency": Latency,
             "caption_latency": CaptionLatency, "retries": RetryInfo,
             "per_style": PerStyle, "per_clip": PerClip, "records": Records},
            f, ensure_ascii=False, indent=2,
        )

    # Human summary -> text (printed AND appended).
    Report = FormatSummary(
        RootConfig, TestConfig, Timestamp, Records, Overall, PerStyle, PerClip,
        ElapsedSeconds, Latency, CaptionLatency, RetryInfo
    )
    print("\n" + Report)
    with open(Paths["ReportText"], "a", encoding="utf-8") as f:
        f.write("\n" + Report + "\n")


def FormatSummary(RootConfig, TestConfig, Timestamp, Records, Overall, PerStyle, PerClip,
                  ElapsedSeconds, Latency, CaptionLatency, RetryInfo) -> str:
    Bar = "=" * ReportWidth
    Rule = "-" * ReportWidth
    L: list[str] = []
    L.append(Bar)
    L.append(f" TEST RUN   {Timestamp}   |   Total time: {FormatDuration(ElapsedSeconds)}")
    L.append(Bar)
    bEnsemble = len(RootConfig.Models) > 1
    L.append(" CAPTION PIPELINE (root config.yaml)")
    L.append(f"   mode: {'ensemble' if bEnsemble else 'single-model'}   |   candidate models: {len(RootConfig.Models)}")
    for i, m in enumerate(RootConfig.Models, 1):
        L.append(
            f"   model {i}: {m.Id}"
        )
        L.append(
            f"            provider={m.Provider}, temp={m.Temperature}, "
            f"max_tokens={m.MaxTokens}, reasoning={m.ReasoningEffort}"
        )
    if bEnsemble:
        j = RootConfig.Judge
        L.append(f"   judge  : {j.Id}")
        L.append(
            f"            provider={j.Provider}, temp={j.Temperature}, "
            f"max_tokens={j.MaxTokens}, reasoning={j.ReasoningEffort}, pass_frames={j.PassFrames}"
        )
    else:
        L.append("   judge  : (disabled — single-model mode)")
    L.append(f"   style temperatures: {RootConfig.StyleTemperatures}")
    Fr = RootConfig.Frames
    L.append(
        f"   frames : {Fr.PerThirtySeconds}/30s, max {Fr.MaxTotal}, width {Fr.Width}px, "
        f"q{Fr.JpegQuality}, source={Fr.Source}, fallback={Fr.EnableFallback}"
    )
    L.append("")

    Sc = TestConfig["Scorer"]
    W = TestConfig["Weights"]
    L.append(" SCORER (Testing/test_config.yaml)")
    L.append(f"   model  : {Sc['Id']}")
    L.append(
        f"            provider={Sc['Provider']}, temp={Sc['Temperature']}, "
        f"max_tokens={Sc['MaxTokens']}, reasoning={Sc['ReasoningEffort']}, "
        f"pass_frames={Sc.get('PassFrames', True)}"
    )
    L.append(f"   weights: accuracy={W['Accuracy']}, style_match={W['StyleMatch']}")
    L.append(f"   captions scored: {len(Records)}")
    L.append("")

    # Headline numbers.
    L.append(" OVERALL")
    L.append(f"   accuracy    : {Overall['accuracy']:.3f}")
    L.append(f"   style_match : {Overall['style_match']:.3f}")
    L.append(f"   final score : {Overall['score']:.3f}")
    L.append("")

    # Per-FINAL-CAPTION latency: candidates (run concurrently) + judge, combined. This is the
    # number that matters if "per request" means per caption. The 30s marker flags the cap.
    Cl = CaptionLatency
    CapFlag = "  <-- OVER 30s!" if Cl["max"] >= 30.0 else ("  <-- near 30s" if Cl["max"] >= 25.0 else "")
    L.append(" PER CAPTION LATENCY (concurrent candidates + judge, end to end)")
    L.append(f"   captions: {Cl['count']}   avg {Cl['avg']:.1f}s   min {Cl['min']:.1f}s   "
             f"max {Cl['max']:.1f}s   (30s cap){CapFlag}")
    L.append("")

    # Per-request latency (captioning phase: every candidate + judge API call). THIS is the
    # number that maps to the hackathon's "response time per request < 30s" rule.
    Ov = Latency["overall"]
    L.append(" PER REQUEST LATENCY (single candidate/judge API call -- maps to the 30s rule)")
    L.append(f"   requests: {Ov['count']}   avg {Ov['avg']:.1f}s   min {Ov['min']:.1f}s   "
             f"max {Ov['max']:.1f}s   (30s cap)")
    if Latency["per_model"]:
        L.append("   by model                     n     avg      min      max")
        for ModelName, St in Latency["per_model"].items():
            Flag = "  <-- near 30s cap" if St["max"] >= 25.0 else ""
            L.append(f"   {ModelName:<24} {St['count']:>4}   {St['avg']:5.1f}s   "
                     f"{St['min']:5.1f}s   {St['max']:5.1f}s{Flag}")
    L.append("")

    # Retries: explains any inflated PER CAPTION time. 0 retries => per-caption time is pure
    # model+throttle; any retry adds >=6.5s backoff to that one caption (not to any request).
    Rt = RetryInfo
    L.append(" RETRIES / FAILURES (why a per-caption time may exceed its request times)")
    L.append(f"   requests that retried then succeeded: {Rt['retried_requests']}   "
             f"failed after all attempts: {Rt['failed_requests']}   "
             f"max attempts on any request: {Rt['max_attempts']}")
    if Rt["by_model"]:
        for ModelName, Rc in Rt["by_model"].items():
            L.append(f"   {ModelName:<24} retried={Rc['retried']}  failed={Rc['failed']}")
    elif Rt["retried_requests"] == 0 and Rt["failed_requests"] == 0:
        L.append("   none - every request succeeded on the first attempt")
    L.append("")

    # Per-style table.
    L.append(" PER STYLE                     n   accuracy   style   score")
    for Style, Agg in PerStyle.items():
        L.append(f"   {Style:<24} {Agg['n']:>3}    {Agg['accuracy']:.3f}    {Agg['style_match']:.3f}   {Agg['score']:.3f}")
    L.append("")

    # Per-clip score (sorted best -> worst).
    L.append(" PER CLIP (mean score, best -> worst)")
    for Clip, Score in sorted(PerClip.items(), key=lambda kv: kv[1], reverse=True):
        L.append(f"   {Clip:<24} {Score:.3f}")
    L.append("")

    # Full detail: every caption with its two scores.
    L.append(" DETAIL")
    for Clip in sorted({r["task_id"] for r in Records}):
        L.append(f"  {Clip}")
        for r in [x for x in Records if x["task_id"] == Clip]:
            L.append(f"     [{r['style']}]  acc={r['accuracy']:.2f}  style={r['style_match']:.2f}  score={r['score']:.2f}")
            L.append(f"        {r['caption']}")
    L.append(Bar)
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
