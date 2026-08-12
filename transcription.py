from io import BytesIO
from pathlib import Path
from functools import cache
from difflib import SequenceMatcher
from typing import Any, ClassVar, Concatenate, cast
from collections.abc import Callable, Iterator

from PIL import Image
from jinja2 import Environment, FileSystemLoader, Template
from ollama import AsyncClient
import ollama
from paddleocr import PaddleOCR  # pyright: ignore[reportMissingTypeStubs]
from pydantic import BaseModel, ConfigDict, RootModel
import re

from constants import (
    DEFAULT_OPTIONS,
    EXAMPLE_CERTAINTY_THRESHOLD,
    FINANCIAL_ARTIFACT_WIDTH,
    FINANCIAL_CROP_MARGIN_X,
    FINANCIAL_CROP_MARGIN_Y,
    FINANCIAL_CROP_SCALE,
    FINANCIAL_NARROW_VSEP_MULTIPLIER,
    FINANCIAL_WIDE_VSEP_MULTIPLIER,
    LINE_GROUPING_HEIGHT_MULTIPLIER,
    MODEL,
    REVIEW_CERTAINTY_THRESHOLD,
    SINGLE_GLYPH_WIDTH,
    VSEP_REVIEW_THRESHOLD,
)
from models import CheckEvent, OCRLineGroup, OCRText


ollama_client = AsyncClient()
MONEY_PATTERN = re.compile(r"\$|(?<!\d)\d+\.\d{2}(?!\d)")


def load_jinja_template(template: str) -> Template:
    file_loader = FileSystemLoader("prompts")
    env = Environment(
        loader=file_loader,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    return env.get_template(template)


@cache
def _paddle_ocr() -> PaddleOCR:
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


class PaddleOCRResult(BaseModel):
    rec_texts: list[str]
    rec_scores: list[float]
    rec_boxes: list[list[int]]


class PaddleOcrPredictResult(RootModel[list[PaddleOCRResult]]): ...


def png_ocr(png_path: Path) -> list[OCRText]:
    p_ocr = _paddle_ocr()
    paddle_result: list[object] = cast(list[object], p_ocr.predict(str(png_path)))  # pyright: ignore[reportUnknownMemberType]
    predictions = PaddleOcrPredictResult.model_validate(paddle_result).root
    results: list[OCRText] = []
    for prediction in predictions:
        for text, confidence, box in zip(
            prediction.rec_texts,
            prediction.rec_scores,
            prediction.rec_boxes,
        ):
            results.append(
                OCRText(
                    id=len(results),
                    text=text,
                    confidence=confidence,
                    bounding_box=(box[0], box[1], box[2], box[3]),
                )
            )

    return results


def ocr_checker[**P](
    func: Callable[Concatenate[OCRText, P], bool],
) -> Callable[Concatenate[OCRText, P], bool]:
    def wrapper(ocr_text: OCRText, *args: P.args, **kwargs: P.kwargs) -> bool:
        result = func(ocr_text, *args, **kwargs)
        ocr_text.history.append(CheckEvent(func.__name__, result))
        return result

    return wrapper


@ocr_checker
def is_financial_artifact_width(record: OCRText, image_width: int) -> bool:
    return (record.width / image_width) < FINANCIAL_ARTIFACT_WIDTH


@ocr_checker
def is_wide_financial_artifact(record: OCRText, image_width: int) -> bool:
    return (record.width / image_width) >= SINGLE_GLYPH_WIDTH


@ocr_checker
def vertical_separation_check(
    record: OCRText,
    other: OCRText,
    separation_multiplier: float = VSEP_REVIEW_THRESHOLD,
) -> bool:
    scale = max(abs(record.height), abs(other.height))
    return abs(record.center[1] - other.center[1]) / scale <= separation_multiplier


@ocr_checker
def is_uncertain_financial_artifact(
    record: OCRText, records: list[OCRText], image_width: int
) -> bool:
    """
    small + low-confidence fragment
    +
    vertical alignment with money or a financial label
    =
    candidate uncertain financial artifact
    """
    if record.text == "$" or record.confidence >= REVIEW_CERTAINTY_THRESHOLD:
        return False

    if not is_financial_artifact_width(record, image_width):
        return False

    is_wide_artifact = is_wide_financial_artifact(record, image_width)
    if is_wide_artifact and not (
        "$" in record.text and any(character.isdigit() for character in record.text)
    ):
        return False

    separation_multiplier = (
        FINANCIAL_WIDE_VSEP_MULTIPLIER
        if is_wide_artifact
        else FINANCIAL_NARROW_VSEP_MULTIPLIER
    )

    for other in records:
        if other is record:
            continue
        if MONEY_PATTERN.search(other.text) and vertical_separation_check(
            record, other, separation_multiplier
        ):
            return True

    return False


def record_crop(image: Image.Image, record: OCRText):
    """
    Crop doc image to only contain OCRText + margin
    """
    crop_left = max(record.left - FINANCIAL_CROP_MARGIN_X, 0)
    crop_top = max(record.top - FINANCIAL_CROP_MARGIN_Y, 0)
    crop_right = min(record.right + FINANCIAL_CROP_MARGIN_X, image.width)
    crop_bottom = min(record.bottom + FINANCIAL_CROP_MARGIN_Y, image.height)
    crop = image.crop((crop_left, crop_top, crop_right, crop_bottom)).resize(
        (
            (crop_right - crop_left) * FINANCIAL_CROP_SCALE,
            (crop_bottom - crop_top) * FINANCIAL_CROP_SCALE,
        )
    )
    img_bytes = BytesIO()
    crop.save(img_bytes, format="PNG")
    return img_bytes.getvalue()


class OllamaChunk(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(from_attributes=True)

    response: str


def stream_ollama_response(
    prompt: str,
    images: list[bytes] | None = None,
    options: dict[str, Any] = DEFAULT_OPTIONS,
    think: bool | None = False,
) -> Iterator[OllamaChunk]:
    response_stream = ollama.generate(
        model=MODEL,
        prompt=prompt,
        images=images,
        options=options,
        think=think,
        stream=True,
    )
    for chunk in response_stream:
        yield OllamaChunk.model_validate(chunk)


async def ollama_response_async(
    prompt: str,
    images: list[bytes] | None = None,
    options: dict[str, Any] = DEFAULT_OPTIONS,
    think: bool | None = False,
    response_format: dict[str, Any] | None = None,
) -> str:
    response = await ollama_client.generate(
        model=MODEL,
        prompt=prompt,
        images=images,
        options=options,
        think=think,
        format=response_format,
        stream=False,
    )
    return OllamaChunk.model_validate(response).response


async def resolve_wide_artifact(record: OCRText, image: Image.Image):
    cropped_image = record_crop(image, record)
    prompt = load_jinja_template("resolve_financial_fragment.j2j").render()
    response = await ollama_response_async(prompt=prompt, images=[cropped_image])
    record.update_text(response, resolve_wide_artifact.__name__)


FINANCIAL_SYMBOLS = frozenset({"$", "@", "^", "%", "*", "&", "#"})


@ocr_checker
def is_symbol_example(record: OCRText) -> bool:
    return record.text.strip() in FINANCIAL_SYMBOLS


async def resolve_financial_glyph(
    record: OCRText, records: list[OCRText], image: Image.Image
) -> None:
    cropped_image = record_crop(image, record)
    examples = sorted(
        (
            other
            for other in records
            if other is not record
            and is_symbol_example(other)
            and other.confidence >= EXAMPLE_CERTAINTY_THRESHOLD
        ),
        key=lambda x: abs(
            (x.center[0] - record.center[0]) ** 2
            + (x.center[1] - record.center[1]) ** 2
        ),
    )[:2]

    example_text = ", ".join(
        f"image {index} is verified {example.text!r}"
        for index, example in enumerate(examples, 1)
    )
    template = load_jinja_template("resolve_financial_glyph.j2j")
    prompt = template.render(example_text=example_text)
    images = [record_crop(image, example) for example in examples]
    images.append(cropped_image)
    response = await ollama_response_async(prompt=prompt, images=images)
    record.update_text(response, resolve_financial_glyph.__name__)


@ocr_checker
def is_uncertain_text(record: OCRText) -> bool:
    text = record.text
    embedded_digit = any(
        character.isdigit()
        and 0 < index < len(text) - 1
        and text[index - 1].isalpha()
        and text[index + 1].isalpha()
        for index, character in enumerate(text)
    )
    suspicious_measurement = re.search(
        r"""
        \b                  # start at a word boundary
        \d+                 # measurement value
        [A-Za-z]            # suspicious character between the value and unit
        (?=                 # require a known unit immediately after it
            (?:GRAM|KG|G|ML|L)
            \b
        )
        """,
        text,
        re.IGNORECASE | re.VERBOSE,
    )
    suspicious_token_ending = re.search(
        r"""
        [A-Za-z]            # alphabetic token content
        \d                  # likely letter-to-digit OCR substitution
        \b                  # digit must end the token, as in "Ful1"
        """,
        text,
        re.VERBOSE,
    )
    return (
        any(ord(character) > 127 for character in text)
        or embedded_digit
        or bool(suspicious_measurement)
        or bool(suspicious_token_ending)
    )


async def resolve_uncertain_text(record: OCRText, image: Image.Image) -> None:
    prompt = load_jinja_template("resolve_uncertain_text.j2j").render(
        original_text=record.text
    )
    response = await ollama_response_async(
        prompt=prompt,
        images=[record_crop(image, record)],
        think=False,
        options={"temperature": 0, "num_ctx": 4096},
    )

    # Normalise spacing
    candidate = " ".join(response.split())

    # Reject broad rewrites or empty strings
    if (
        not candidate
        or abs(len(record.text) - len(candidate)) > 2
        or SequenceMatcher(None, record.text, candidate).ratio() < 0.9
    ):
        return

    record.update_text(candidate, resolve_uncertain_text.__name__)


async def transcribe_receipt(png_path: Path) -> list[OCRText]:
    """Run Paddle once, then resolve only records selected for review."""
    records = png_ocr(png_path)

    with Image.open(png_path) as source_image:
        image = source_image.convert("RGB")
        for record in records:
            if is_uncertain_financial_artifact(record, records, image.width):
                if is_wide_financial_artifact(record, image.width):
                    await resolve_wide_artifact(record, image)
                else:
                    await resolve_financial_glyph(record, records, image)
            elif is_uncertain_text(record):
                await resolve_uncertain_text(record, image)

    return sorted(records, key=lambda record: (record.top, record.left))


def write_transcription(records: list[OCRText]) -> str:
    lines: list[OCRLineGroup] = []

    for record in sorted(records, key=lambda item: (item.top, item.left)):
        if lines:
            current_line = lines[-1]
            vsep = abs(record.center[1] - current_line.center)
            if vsep <= current_line.height * LINE_GROUPING_HEIGHT_MULTIPLIER:
                current_line.records.append(record)
                continue
        lines.append(OCRLineGroup(records=[record]))

    text_lines = [
        "  ".join(
            item.text for item in sorted(line.records, key=lambda item: item.left)
        )
        for line in lines
    ]
    return "\n".join(text_lines) + ("\n" if text_lines else "")
