import os
from dataclasses import dataclass

import yaml


@dataclass
class FModelConfig:
    Provider: str
    Id: str
    BaseUrl: str
    Temperature: float
    MaxTokens: int
    ReasoningEffort: str


@dataclass
class FFramesConfig:
    PerThirtySeconds: int
    MaxTotal: int
    Width: int
    JpegQuality: int
    Strategy: str
    MaxPayloadMB: float
    Source: str
    EnableFallback: bool


@dataclass
class FClientConfig:
    TimeoutSeconds: int
    MaxRetries: int
    BackoffSeconds: float


@dataclass
class FRuntimeConfig:
    MaxWorkers: int


@dataclass
class FPathsConfig:
    Input: str
    Output: str
    WorkDir: str


@dataclass
class FConfig:
    Model: FModelConfig
    Frames: FFramesConfig
    Client: FClientConfig
    Runtime: FRuntimeConfig
    Paths: FPathsConfig
    DefaultStyles: list[str]
    StyleTemperatures: dict[str, float]


def SelectPaths(RawPaths: dict) -> dict:
    # Docker always creates /.dockerenv in a container; use the Docker paths there, Local otherwise.
    bInDocker: bool = os.path.exists("/.dockerenv")
    return RawPaths["Docker"] if bInDocker else RawPaths["Local"]


def LoadConfig(ConfigPath: str = "config.yaml") -> FConfig:
    with open(ConfigPath, "r", encoding="utf-8") as ConfigFile:
        RawConfig: dict = yaml.safe_load(ConfigFile)

    return FConfig(
        Model=FModelConfig(**RawConfig["Model"]),
        Frames=FFramesConfig(**RawConfig["Frames"]),
        Client=FClientConfig(**RawConfig["Client"]),
        Runtime=FRuntimeConfig(**RawConfig["Runtime"]),
        Paths=FPathsConfig(**SelectPaths(RawConfig["Paths"])),
        DefaultStyles=RawConfig["DefaultStyles"],
        StyleTemperatures=RawConfig["StyleTemperatures"],
    )
