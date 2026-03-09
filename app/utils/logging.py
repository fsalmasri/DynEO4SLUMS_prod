import logging

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%d:%m:%Y %H:%M:%S",
    )


