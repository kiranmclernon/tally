import logging
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from paddleocr import PaddleOCR
from models import Config, OCRText
from constants import CONFIG_NAME, DEFAULT_OPTIONS, MODEL
from pydantic import ValidationError
import yaml
from collections.abc import Iterator, Sequence
import ollama
from ollama import GenerateResponse
from jinja2 import Template, FileSystemLoader, Environment
import subprocess

from ocr import png_ocr


def load_config() -> Config:
    with open(CONFIG_NAME, "r") as f:
        config_yaml = yaml.safe_load(f)  # pyright: ignore[reportAny]
        try:
            return Config(**config_yaml)  # pyright: ignore[reportAny]

        except ValidationError as e:
            logging.error(f"Error loading config: {e}")
            raise e


def load_jinja_template(template: str) -> Template:
    file_loader = FileSystemLoader("prompts")
    env = Environment(
        loader=file_loader,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(template)


def stream_ollama_response(
    prompt: str,
    images: list[bytes] | None = None,
    options: dict[str, Any] = DEFAULT_OPTIONS,  # pyright: ignore[reportExplicitAny]
    think: bool | None = False,
    stream: bool = True,
) -> Iterator[GenerateResponse]:
    return ollama.generate(  # pyright: ignore[reportReturnType]
        model=MODEL,
        prompt=prompt,
        images=images,
        options=options,
        think=think,
        stream=stream,
    )


def toggle_ollama(action: Literal["start", "stop"]):
    logging.info(f"Toggling Ollama")
    if action == "start":
        _ = subprocess.run(
            ["brew", "services", "start", "ollama"], stdout=subprocess.DEVNULL
        )
    else:
        _ = subprocess.run(
            ["brew", "services", "stop", "ollama"], stdout=subprocess.DEVNULL
        )


class _OCRResult(TypedDict):
    rec_texts: Sequence[str]
    rec_scores: Sequence[float]
    rec_boxes: Sequence[Sequence[int]]


def png_ocr(png_path: Path) -> tuple[list[object], list[OCRText]]:
    ocr = PaddleOCR(
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )
    raw_results = cast(
        list[object],
        ocr.predict(str(png_path)),  # pyright: ignore[reportUnknownMemberType]
    )

    ocr_records: list[OCRText] = []
    for raw_result in raw_results:
        result = cast(_OCRResult, raw_result)
        records = zip(
            result["rec_texts"],
            result["rec_scores"],
            result["rec_boxes"],
            strict=True,
        )
        for text, confidence, box in records:
            bounding_box = tuple(int(coordinate) for coordinate in box)
            if len(bounding_box) != 4:
                raise ValueError(f"Expected four box coordinates, got {bounding_box!r}")

            ocr_records.append(
                OCRText(
                    text=text,
                    confidence=confidence,
                    bounding_box=bounding_box,
                )
            )

    return raw_results, ocr_records


def transcribe_receipt(run_id: str, receipt_image_path: Path, output_dir: Path):
    raw_result, result = png_ocr(receipt_image_path)
    prompt = load_jinja_template("transcription.j2j").render(ocr_evidence=result)
    logging.info(f"Streaming Ollama Response")
    response = stream_ollama_response(
        prompt=prompt,
        images=[receipt_image_path.read_bytes()],
        think=False,
        stream=True,
    )
    response_chunks: list[str] = []
    for chunk in response:
        if chunk_response := chunk.response:
            response_chunks.append(chunk_response)
    transcription = "".join(response_chunks)

    artifact_path = output_dir / receipt_image_path.stem / run_id

    artifact_path.mkdir(parents=True, exist_ok=True)

    with open(artifact_path / f"{receipt_image_path.stem}-transcription.txt", "w") as f:
        f.write(transcription)

    for idx, res in enumerate(raw_result):
        res.save_to_img(artifact_path / f"output-{idx}")
