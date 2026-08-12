import argparse
import asyncio
import logging
from pathlib import Path

from transcription import transcribe_receipt, write_transcription


class Arguments(argparse.Namespace):
    input_image: Path
    output: Path

    def __init__(self) -> None:
        super().__init__()
        self.input_image = Path()
        self.output = Path()


def parse_args() -> Arguments:
    parser = argparse.ArgumentParser(description="Transcribe a receipt image.")
    _ = parser.add_argument("input_image", type=Path, help="Path to the receipt image")
    _ = parser.add_argument("output", type=Path, help="Path for the text transcription")
    args = parser.parse_args(namespace=Arguments())
    if not args.input_image.is_file():
        parser.error(f"input image does not exist: {args.input_image}")
    return args


def main() -> None:
    args = parse_args()
    records = asyncio.run(transcribe_receipt(args.input_image))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _ = args.output.write_text(write_transcription(records), encoding="utf-8")
    logging.info("Wrote transcription to %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
