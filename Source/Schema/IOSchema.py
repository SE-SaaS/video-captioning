import json
import os

from Source.Schema.Models import ECaptionStyle, FCaptionResult, FVideoTask


def ParseStyles(RawStyles: object, TaskId: str) -> list[ECaptionStyle]:
    if not isinstance(RawStyles, list) or not RawStyles:
        raise ValueError(f"Task '{TaskId}' has no styles to caption.")

    Styles: list[ECaptionStyle] = []
    for RawStyle in RawStyles:
        Style: ECaptionStyle = ECaptionStyle.FromValue(RawStyle)
        if Style not in Styles:
            Styles.append(Style)
    return Styles


def LoadTasks(InputPath: str, DefaultStyles: list[str]) -> list[FVideoTask]:
    with open(InputPath, "r", encoding="utf-8") as InputFile:
        RawTasks: object = json.load(InputFile)

    if not isinstance(RawTasks, list):
        raise ValueError("tasks.json must be a JSON array of task objects.")

    Tasks: list[FVideoTask] = []
    for TaskIndex, RawTask in enumerate(RawTasks):
        if not isinstance(RawTask, dict):
            raise ValueError(f"Task at index {TaskIndex} is not an object.")

        TaskId: object = RawTask.get("task_id")
        VideoUrl: object = RawTask.get("video_url")
        if not isinstance(TaskId, str) or not TaskId:
            raise ValueError(f"Task at index {TaskIndex} is missing a valid 'task_id'.")
        if not isinstance(VideoUrl, str) or not VideoUrl:
            raise ValueError(f"Task '{TaskId}' is missing a valid 'video_url'.")

        RawStyles: object = RawTask.get("styles") or DefaultStyles
        Styles: list[ECaptionStyle] = ParseStyles(RawStyles, TaskId)
        Tasks.append(FVideoTask(TaskId=TaskId, VideoUrl=VideoUrl, Styles=Styles))

    return Tasks


def MissingStyles(Task: FVideoTask, Result: FCaptionResult) -> list[ECaptionStyle]:
    return [Style for Style in Task.Styles if not Result.Captions.get(Style.value)]


def WriteResults(OutputPath: str, Results: list[FCaptionResult]) -> None:
    OutputDir: str = os.path.dirname(OutputPath)
    if OutputDir:
        os.makedirs(OutputDir, exist_ok=True)

    Payload: list[dict] = [
        {"task_id": Result.TaskId, "captions": Result.Captions} for Result in Results
    ]
    with open(OutputPath, "w", encoding="utf-8") as OutputFile:
        json.dump(Payload, OutputFile, ensure_ascii=False, indent=2)
