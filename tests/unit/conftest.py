from __future__ import annotations

import sys
from pathlib import Path

SOURCE = Path(__file__).parents[2] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
