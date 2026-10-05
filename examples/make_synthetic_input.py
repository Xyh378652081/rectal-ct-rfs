"""Regenerate the non-patient synthetic input supplied with the release."""

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts" / "generate_public_assets.py"), run_name="__main__")
