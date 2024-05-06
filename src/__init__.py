# add package directory in system path, so submodules can easily be accessed
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.as_posix()
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
