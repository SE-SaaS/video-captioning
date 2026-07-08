from dataclasses import dataclass
from enum import Enum


class ECaptionStyle(Enum):
    # Values are the exact keys the harness scores against. Do not rename.
    Formal = "formal"
    Sarcastic = "sarcastic"
    HumorousTech = "humorous_tech"
    HumorousNonTech = "humorous_non_tech"

    @classmethod
    def FromValue(cls, StyleValue: str) -> "ECaptionStyle":
        try:
            return cls(StyleValue)
        except ValueError:
            ValidValues: str = ", ".join(Style.value for Style in cls)
            raise ValueError(
                f"Unknown style '{StyleValue}'. Expected one of: {ValidValues}."
            )


@dataclass
class FVideoTask:
    TaskId: str
    VideoUrl: str
    Styles: list[ECaptionStyle]


@dataclass
class FCaptionResult:
    TaskId: str
    Captions: dict[str, str]
