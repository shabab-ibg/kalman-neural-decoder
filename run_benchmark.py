"""Run the models that are actually implemented in this checkout.

This runner intentionally records only freshly computed metrics. Do not add
historical scores for models that this script did not execute: that makes
results.csv look like a reproducible benchmark when it is not.

The current repository checkout must include src/data_pipeline.py and
src/classic_models.py, plus data/mc_maze_20ms.npz, before this script can run.
"""
from pathlib import Path
import os

import pandas as pd
import numpy as np
import torch

DATA_PATH = Path("data/mc_maze_20ms.npz")
RESULTS_PATH = Path("results.csv")


def main():
    # Check the repository is complete before importing project modules.
    required_modules = [
        Path("src/data_pipeline.py"),
        Path("src/classic_models.py"),
    ]
    missing = [str(path) for path in required_modules if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "This checkout is missing required project source files: "
            + ", ".join(missing)
            + ". Restore these modules before running the benchmark. "
            "No metrics were generated."
        )
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Missing dataset at {DATA_PATH}. Place the preprocessed MC_Maze "
            "file there before running the benchmark. No metrics were generated."
        )

    # Imports occur after validation so missing files produce a useful message.
    from src.data_pipeline import NLBMazePipeline
    from src.classic_models import SingleLagKalmanDecoder, compute_r2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running the implemented Kalman decoder on: {device}")

    pipeline = NLBMazePipeline(str(DATA_PATH), history_len=10, device=device)
    decoder = SingleLagKalmanDecoder(lag_bins=6)
    decoder.fit(
        pipeline.X, pipeline.Y, pipeline.starts, pipeline.ends, pipeline.tr_s
    )
    y_true, y_pred = decoder.decode_sequence(
        pipeline.X,
        pipeline.Y,
        pipeline.starts,
        pipeline.ends,
        pipeline.test_segs,
    )
    r2 = np.asarray(compute_r2(y_true, y_pred), dtype=float).reshape(-1)
    if r2.size < 2 or not np.isfinite(r2[:2]).all():
        raise ValueError(
            "compute_r2 must return finite per-axis R² values for vx and vy."
        )

    # Only include results computed in this run. Other reported model scores
    # are deliberately excluded until their implementations are executable here.
    df = pd.DataFrame(
        [{
            "model": "Kalman (plain, linear observation)",
            "R2_vx": float(r2[0]),
            "R2_vy": float(r2[1]),
            "R2_mean": float(np.mean(r2[:2])),
            "status": "computed_this_run",
        }]
    )
    df.to_csv(RESULTS_PATH, index=False)
    print(df.to_string(index=False))
    print(f"Saved freshly computed metrics to {RESULTS_PATH}.")
    print(
        "Note: this runner evaluates only the Kalman decoder. Wiener, hybrid, "
        "and GRU results are not reproduced by this script."
    )


if __name__ == "__main__":
    main()
