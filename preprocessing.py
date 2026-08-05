import fitz
from fitz import Page
from pathlib import Path
import logging
from PIL import Image
from pillow_heif import register_heif_opener
from constants import PNG_CONVERSION_DPI

register_heif_opener()


def pdf_to_png(pdf_path: Path, output_dir: Path):
    doc = fitz.open(pdf_path)

    logging.debug(f"Converting {pdf_path} to png")
    logging.debug(f"Converting {pdf_path} to png")

    page: Page = doc.load_page(0)

    pix = page.get_pixmap(dpi=PNG_CONVERSION_DPI)

    output_file_path = output_dir / f"{pdf_path.name}.png"

    pix.save(output_file_path)

    logging.debug(f"Converted {pdf_path} to png")


def heic_to_png(heic_path: Path, output_dir: Path):
    img = Image.open(heic_path)

    logging.debug(f"Converting {heic_path} to png")

    output_file_path = output_dir / f"{heic_path.name}.png"

    img.save(output_file_path, dpi=(PNG_CONVERSION_DPI, PNG_CONVERSION_DPI))

    logging.debug(f"Converted {pdf_path} to png")
