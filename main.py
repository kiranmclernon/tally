import logging
from pathlib import Path
from time import sleep
import uuid
from utils import (
    toggle_ollama,
    transcribe_receipt,
)
import sys


path = Path(
    "groceries/validation/tuning-images"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
    force=True,
)


def main():
    toggle_ollama("start")
    sleep(5)

    run_id = uuid.uuid4()

    for idx, image_path in enumerate(path.iterdir()):
        logging.info(f"Begining {idx}: {image_path.stem}")
        transcribe_receipt(str(run_id), image_path, Path("output"))

    toggle_ollama("stop")


if __name__ == "__main__":
    main()
