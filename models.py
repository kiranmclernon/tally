from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class Config(BaseModel):
    img_dir: Path
    transcription_dir: Path


class OCRText(BaseModel):
    text: str
    confidence: float
    bounding_box: tuple[int, int, int, int]
