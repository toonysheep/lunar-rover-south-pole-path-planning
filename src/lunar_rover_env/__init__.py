"""Standalone access to the archived lunar rover environment."""
from pathlib import Path
import sys

VENDOR = Path(__file__).resolve().parents[2] / "vendor"
if "gym" in sys.modules and not Path(sys.modules["gym"].__file__).resolve().is_relative_to(VENDOR):
    raise RuntimeError("Import lunar_rover_env before any other gym installation; use a fresh Python process.")
sys.path.insert(0, str(VENDOR))
from .environment import LunarRoverEnv

__all__ = ["LunarRoverEnv"]
