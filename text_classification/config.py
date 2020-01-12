import logging
import pathlib

logging.basicConfig(level=logging.INFO)

# Dirs
ROOT_DIR = pathlib.Path(__file__).parent.absolute()
DUMP_DIR = ROOT_DIR / 'dumps'
