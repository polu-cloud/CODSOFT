import logging
import os


def setup_logging():
    os.makedirs("outputs", exist_ok=True)
    logging.basicConfig(
        filename="outputs/pipeline.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
