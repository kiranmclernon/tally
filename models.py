from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from functools import cached_property


class Config(BaseModel):
    img_dir: Path
    transcription_dir: Path


@dataclass
class OCRText:
    id: int
    text: str
    confidence: float
    bounding_box: tuple[int, int, int, int]
    history: list["HistoryEvent"] = field(default_factory=list)

    def update_text(self, value: str, operation: str):
        if value != self.text:
            self.history.append(StateTransition(self.text, value, operation))
            self.text = value

    @cached_property
    def left(self):
        return self.bounding_box[0]

    @cached_property
    def top(self):
        return self.bounding_box[1]

    @cached_property
    def right(self):
        return self.bounding_box[2]

    @cached_property
    def bottom(self):
        return self.bounding_box[3]

    @cached_property
    def height(self):
        return self.top - self.bottom

    @cached_property
    def center(self):
        return (self.right + self.left) / 2, (self.top + self.bottom) / 2

    @cached_property
    def width(self):
        return self.right - self.left


@dataclass
class StateTransition:
    kind: Literal["state_transition"] = field(default="state_transition", init=False)
    before: str
    after: str
    operation: str


@dataclass
class CheckEvent:
    kind: Literal["check"] = field(default="check", init=False)
    check: str
    result: bool


type HistoryEvent = CheckEvent | StateTransition


@dataclass
class OCRLineGroup:
    """OCR records that occupy the same visual receipt line."""

    records: list[OCRText]

    @property
    def height(self) -> int:
        return max(abs(record.height) for record in self.records)

    @property
    def center(self) -> float:
        return sum(record.center[1] for record in self.records) / len(self.records)
