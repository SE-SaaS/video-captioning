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


def LoadConfig(ConfigPath: str = "config.yaml") -> FConfig:
    with open(ConfigPath, "r", encoding="utf-8") as ConfigFile:
        RawConfig: dict = yaml.safe_load(ConfigFile)

    return FConfig(
        Model=FModelConfig(**RawConfig["Model"]),
        Frames=FFramesConfig(**RawConfig["Frames"]),
        Client=FClientConfig(**RawConfig["Client"]),
        Runtime=FRuntimeConfig(**RawConfig["Runtime"]),
        Paths=FPathsConfig(**RawConfig["Paths"]),
        DefaultStyles=RawConfig["DefaultStyles"],
    )
