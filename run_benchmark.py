from pathlib import Path

import pandas as pd
import numpy as np
import torch

from src.data.classic_models import SingleLagKalmanDecoder, compute_r2
from src.data.data_pipeline import NLBMazePipeline


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmark on: {device}")

    repo_root = Path(__file__).resolve().parent
    dataset_path = repo_root / "src" / "data" / "mc_maze_20ms.npz"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Expected src/data/mc_maze_20ms.npz."
        )

    pipeline = NLBMazePipeline(
        file_path=dataset_path,
        history_len=10,
        device=device,
    )

    # State-space Kalman decoder.
    kf = SingleLagKalmanDecoder(lag_bins=6)
    kf.fit(
        pipeline.X,
        pipeline.Y,
        pipeline.starts,
        pipeline.ends,
        pipeline.tr_s,
    )
    y_true_kf, y_pred_kf = kf.decode_sequence(
        pipeline.X,
        pipeline.Y,
        pipeline.starts,
        pipeline.ends,
        pipeline.test_segs,
    )
    r2_kf = compute_r2(y_true_kf, y_pred_kf)

    # These non-Kalman metrics are previously reported values, not recomputed
    # by this script. Only the Kalman row is generated from the local run.
    metrics = [
        ("Wiener (ridge, 200 ms)", 0.536, 0.534, 0.535),
        ("Kalman (plain, linear obs.)", r2_kf[0], r2_kf[1], r2_kf.mean()),
        ("Wiener + Kalman smoother", 0.545, 0.545, 0.545),
        ("GRU, windowed", 0.910, 0.869, 0.888),
        ("GRU, full segment history", 0.929, 0.896, 0.915),
    ]

    df = pd.DataFrame(
        metrics,
        columns=["model", "R2_vx", "R2_vy", "R2_mean"],
    )
    print("\n" + "=" * 60 + "\nNEURAL DECODING PERFORMANCE RESULTS\n" + "=" * 60)
    print(df.to_string(index=False))

    output_path = repo_root / "results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nBenchmark complete. Results written to {output_path}")


if __name__ == "__main__":
    main()
