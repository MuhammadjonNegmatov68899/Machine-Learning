import sys
from pathlib import Path

LAB04_ROOT = Path(__file__).resolve().parent.parent
if str(LAB04_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB04_ROOT))
