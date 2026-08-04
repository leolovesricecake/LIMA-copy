"""Repository-root entry point for shared-value word Occlusion and LIME."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mobius.cli.run_first_order import main


if __name__ == "__main__":
    main()
