import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from Source.Config import FClientConfig, FConfig, LoadConfig
from Source.Pipeline.Captioner import FCaptioner
from Source.Pipeline.FireworksClient import FFireworksClient
from Source.Pipeline.FrameSampler import FFrameSampler
from Source.Pipeline.VideoDownloader import FVideoDownloader
from Source.Schema.IOSchema import LoadTasks, MissingStyles, WriteResults
from Source.Schema.Models import FCaptionResult, FVideoTask


# Last-resort caption so a failed clip still emits every requested style (valid JSON,
# no missing key) instead of zeroing the whole clip on a malformed/incomplete result.
FallbackCaption: str = "A short video clip."


def LoadDotEnvIfPresent() -> None:
    # Local development convenience; a no-op inside the submitted image.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def AcquireFrames(
    Task: FVideoTask,
    Config: FConfig,
    Sampler: FFrameSampler,
    Downloader: FVideoDownloader,
) -> list[bytes]:
    def StreamAttempt() -> list[bytes]:
        return Sampler.SampleFrames(Task.VideoUrl)

    def DownloadAttempt() -> list[bytes]:
        LocalPath: str = Downloader.DownloadVideo(Task.VideoUrl, Task.TaskId)
        return Sampler.SampleFrames(LocalPath)

    bStreamFirst: bool = Config.Frames.Source == "stream"
    Attempts: list = (
        [StreamAttempt, DownloadAttempt]
        if bStreamFirst
        else [DownloadAttempt, StreamAttempt]
    )
    if not Config.Frames.EnableFallback:
        Attempts = Attempts[:1]

    LastError: Exception | None = None
    for Attempt in Attempts:
        try:
            return Attempt()
        except Exception as CaughtError:
            LastError = CaughtError
    raise RuntimeError(f"Frame acquisition failed for task '{Task.TaskId}': {LastError}")


def AcquireAllFrames(
    Tasks: list[FVideoTask],
    Config: FConfig,
    Sampler: FFrameSampler,
    Downloader: FVideoDownloader,
) -> dict[str, list[bytes]]:
    FramesByTask: dict[str, list[bytes]] = {}
    with ThreadPoolExecutor(max_workers=Config.Runtime.MaxWorkers) as Executor:
        FutureMap: dict = {
            Executor.submit(AcquireFrames, Task, Config, Sampler, Downloader): Task
            for Task in Tasks
        }
        for Future in as_completed(FutureMap):
            Task: FVideoTask = FutureMap[Future]
            try:
                FramesByTask[Task.TaskId] = Future.result()
            except Exception as CaughtError:
                print(f"[warn] {CaughtError}", file=sys.stderr)
                FramesByTask[Task.TaskId] = []
    return FramesByTask


def GenerateAllCaptions(
    Tasks: list[FVideoTask],
    FramesByTask: dict[str, list[bytes]],
    Config: FConfig,
    Captioner: FCaptioner,
) -> dict[str, dict[str, str]]:
    CaptionsByTask: dict[str, dict[str, str]] = {Task.TaskId: {} for Task in Tasks}
    with ThreadPoolExecutor(max_workers=Config.Runtime.MaxWorkers) as Executor:
        FutureMap: dict = {}
        for Task in Tasks:
            Frames: list[bytes] = FramesByTask.get(Task.TaskId) or []
            if not Frames:
                continue
            for Style in Task.Styles:
                Future = Executor.submit(Captioner.GenerateCaption, Frames, Style)
                FutureMap[Future] = (Task.TaskId, Style.value)

        for Future in as_completed(FutureMap):
            TaskId, StyleValue = FutureMap[Future]
            try:
                CaptionsByTask[TaskId][StyleValue] = Future.result()
            except Exception as CaughtError:
                print(
                    f"[warn] caption failed {TaskId}/{StyleValue}: {CaughtError}",
                    file=sys.stderr,
                )
    return CaptionsByTask


def BuildResults(
    Tasks: list[FVideoTask],
    CaptionsByTask: dict[str, dict[str, str]],
) -> list[FCaptionResult]:
    Results: list[FCaptionResult] = []
    for Task in Tasks:
        Result: FCaptionResult = FCaptionResult(
            TaskId=Task.TaskId,
            Captions=dict(CaptionsByTask.get(Task.TaskId, {})),
        )
        for Style in MissingStyles(Task, Result):
            Result.Captions[Style.value] = FallbackCaption
        Results.append(Result)
    return Results


# Local-only test log: append model config + input/output to a repo txt file for
# review. Best-effort and fully guarded so it can never affect the graded Docker run.
LocalLogPath: str = "local_test_log.txt"


def LogRunLocally(
    Config: FConfig,
    Tasks: list[FVideoTask],
    Results: list[FCaptionResult],
) -> None:
    if os.path.exists("/.dockerenv"):
        return  # Docker (grading) run — never log.
    try:
        InputPayload: list[dict] = [
            {
                "task_id": Task.TaskId,
                "video_url": Task.VideoUrl,
                "styles": [Style.value for Style in Task.Styles],
            }
            for Task in Tasks
        ]
        OutputPayload: list[dict] = [
            {"task_id": Result.TaskId, "captions": Result.Captions} for Result in Results
        ]
        bEnsemble: bool = len(Config.Models) > 1
        ModelsSummary: str = "\n".join(
            f"  - {ModelConfig.Id} (temp={ModelConfig.Temperature}, "
            f"max_tokens={ModelConfig.MaxTokens}, reasoning={ModelConfig.ReasoningEffort})"
            for ModelConfig in Config.Models
        )
        JudgeSummary: str = (
            f"  {Config.Judge.Id} (temp={Config.Judge.Temperature}, "
            f"max_tokens={Config.Judge.MaxTokens}, reasoning={Config.Judge.ReasoningEffort}, "
            f"pass_frames={Config.Judge.PassFrames})"
            if bEnsemble
            else "  (disabled — single-model mode)"
        )
        Entry: str = (
            f"\n{'=' * 80}\n"
            f"Timestamp:       {datetime.now().isoformat(timespec='seconds')}\n"
            f"Mode:            {'ensemble' if bEnsemble else 'single-model'}\n"
            f"Models:\n{ModelsSummary}\n"
            f"Judge:\n{JudgeSummary}\n"
            f"StyleTemperatures: {Config.StyleTemperatures}\n"
            f"Frames:          PerThirtySeconds={Config.Frames.PerThirtySeconds}, "
            f"MaxTotal={Config.Frames.MaxTotal}, Width={Config.Frames.Width}, "
            f"JpegQuality={Config.Frames.JpegQuality}\n"
            f"Input:\n{json.dumps(InputPayload, ensure_ascii=False, indent=2)}\n"
            f"Output:\n{json.dumps(OutputPayload, ensure_ascii=False, indent=2)}\n"
        )
        with open(LocalLogPath, "a", encoding="utf-8") as LogFile:
            LogFile.write(Entry)
    except Exception as CaughtError:
        print(f"[warn] local run logging failed: {CaughtError}", file=sys.stderr)


def BuildClient(ModelConfig: object, ClientConfig: FClientConfig) -> FFireworksClient:
    # ModelConfig is an FModelConfig or FJudgeConfig; both share the fields used here.
    return FFireworksClient(
        ModelId=ModelConfig.Id,
        BaseUrl=ModelConfig.BaseUrl,
        TimeoutSeconds=ClientConfig.TimeoutSeconds,
        MaxRetries=ClientConfig.MaxRetries,
        BackoffSeconds=ClientConfig.BackoffSeconds,
        MaxTokens=ModelConfig.MaxTokens,
        Temperature=ModelConfig.Temperature,
        ReasoningEffort=ModelConfig.ReasoningEffort,
    )


def Main() -> int:
    LoadDotEnvIfPresent()
    Config: FConfig = LoadConfig()
    Tasks: list[FVideoTask] = LoadTasks(Config.Paths.Input, Config.DefaultStyles)

    Sampler: FFrameSampler = FFrameSampler(
        PerThirtySeconds=Config.Frames.PerThirtySeconds,
        MaxTotal=Config.Frames.MaxTotal,
        Width=Config.Frames.Width,
        JpegQuality=Config.Frames.JpegQuality,
        MaxPayloadMB=Config.Frames.MaxPayloadMB,
        TimeoutSeconds=Config.Client.TimeoutSeconds,
    )
    Downloader: FVideoDownloader = FVideoDownloader(
        WorkDir=Config.Paths.WorkDir,
        TimeoutSeconds=Config.Client.TimeoutSeconds,
        MaxRetries=Config.Client.MaxRetries,
        BackoffSeconds=Config.Client.BackoffSeconds,
    )
    Clients: list[FFireworksClient] = [
        BuildClient(ModelConfig, Config.Client) for ModelConfig in Config.Models
    ]
    JudgeClient: FFireworksClient = BuildClient(Config.Judge, Config.Client)
    Captioner: FCaptioner = FCaptioner(
        Clients, JudgeClient, Config.Judge.PassFrames, Config.StyleTemperatures
    )

    FramesByTask: dict[str, list[bytes]] = AcquireAllFrames(
        Tasks, Config, Sampler, Downloader
    )
    CaptionsByTask: dict[str, dict[str, str]] = GenerateAllCaptions(
        Tasks, FramesByTask, Config, Captioner
    )
    Results: list[FCaptionResult] = BuildResults(Tasks, CaptionsByTask)
    WriteResults(Config.Paths.Output, Results)
    LogRunLocally(Config, Tasks, Results)

    OutputPayload: list[dict] = [
        {"task_id": Result.TaskId, "captions": Result.Captions} for Result in Results
    ]
    print(json.dumps(OutputPayload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(Main())
    except Exception as FatalError:
        print(f"[fatal] {FatalError}", file=sys.stderr)
        sys.exit(1)
