"""Run M12 inference on a feature table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rectal_ct_rfs import M12Model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="CSV containing required clinical and imaging features")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV for model predictions")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    prediction = M12Model(args.model_dir).predict(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
