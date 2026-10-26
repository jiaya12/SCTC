"""Compare all per-split results against the paper's recorded outputs."""

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent


def verify(results):
    files = {
        "all_operating_points.csv": [
            "seed",
            "condition",
            "strategy",
            "random_seed",
            "delta",
            "method",
        ],
        "calibration.csv": ["seed", "calibration"],
        "no_current_use.csv": ["seed", "delta", "method"],
    }
    for name, keys in files.items():
        expected = (
            pd.read_csv(ROOT / "reference" / name, dtype={"delta": str})
            .sort_values(keys)
            .reset_index(drop=True)
        )
        actual = (
            pd.read_csv(results / name, dtype={"delta": str})
            .sort_values(keys)
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            expected,
            actual[expected.columns],
            check_dtype=False,
            rtol=1e-10,
            atol=1e-12,
        )
        print(f"{name}: all {len(expected)} rows match")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    verify(parser.parse_args().results)
