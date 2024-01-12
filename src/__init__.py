from pathlib import Path
import logging

ROOT_DIR = Path(__file__).parent.parent.as_posix()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(levelname)s:%(message)s', 
    datefmt='%I:%M:%S'
)
logger = logging.getLogger(__name__)