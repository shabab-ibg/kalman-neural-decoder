# Kalman Neural Decoder

Offline decoding of two-dimensional hand velocity from intracortical motor-cortex spike counts in the Neural Latents Benchmark (NLB) MC_Maze dataset.

This repository contains a single-lag linear Kalman decoder, data preparation utilities, GRU model definitions, and a preprocessed 20 ms MC_Maze dataset. The current benchmark runner evaluates the Kalman decoder only. Historical Wiener, hybrid, and GRU scores are reported separately and are not reproduced by the runner.

> **Scope:** This is offline analysis of an existing intracortical dataset. It is not a real-time BCI, wearable EEG system, or clinical device.

## Reported results

The following values are historical experiment results supplied for this project. Only the Kalman row is recalculated by the current `run_benchmark.py`.

| Decoder | Temporal input | Reported mean test R² |
|---|---|---:|
| Wiener filter (ridge regression) | 200 ms history | 0.535 |
| Single-lag Kalman decoder | Neural observation lag: 120 ms | 0.109 |
| Wiener–Kalman hybrid | 200 ms history | 0.545 |
| Windowed GRU | 200 ms window | 0.888 ± 0.002 |
| Full-segment GRU | Full segment | 0.915 ± 0.002 |

The GRU uncertainties were reported as standard deviations across three random initializations; the underlying per-seed predictions are not included here. The non-Kalman scores are not freshly evaluated by the current runner. Treat this table as a record of reported results, not as a fully reproducible model comparison.

## Dataset and target

- **Dataset:** NLB MC_Maze, preprocessed at 20 ms per bin.
- **Input:** binned intracortical spike counts, loaded as `X`.
- **Target:** planar hand velocity, loaded as `Y`, with two columns: horizontal velocity (v_x) and vertical velocity (v_y).
- **Task:** predict the two-dimensional velocity from lagged neural observations.

The Kalman state is the two-dimensional velocity vector:

$$
x_t = \begin{bmatrix}v_x(t)\\v_y(t)\end{bmatrix}.
$$

There are no additional latent position or acceleration coordinates in this implementation. The observation (z_t) is the vector of neural spike counts across recorded units.

## Kalman model and exact lag convention

The implementation fits a first-order linear transition model to consecutive velocity samples:

$$
x_t = A x_{t-1} + w_t,
$$

and a linear observation model with a fixed neural-to-kinematic delay:

$$
z_{t-k} = Cx_t + d + v_t.
$$

Here, (A) is the (2\times2) velocity transition matrix, (C) maps the two-dimensional velocity state to the neural population, (d) is the observation intercept, and (w_t) and (v_t) are process and observation noise with fitted covariance matrices (Q) and (R).

**Lag convention in the code:** with `lag_bins=k`, the fitting pairs are `X[a:b-k]` with `Y[a+k:b]`. During decoding, at local target index `t`, the filter consumes `X[a+t-k]` to estimate `Y[a+t]`. Thus the neural observation precedes the velocity target by exactly (k) bins. The default is (k=6); at 20 ms per bin this is 120 ms. It is not a six-bin history window: the observation model uses one neural population vector delayed by six bins.

The state is represented internally in mean-centred coordinates. The mean `mu` is estimated from the lagged training targets `Y[a+k:b]`, and is added back to decoded states. The decoder initializes its centred state at zero and its covariance from the fitted training states. The first predictions therefore include an initialization transient.

## Data split and leakage boundary

The preprocessed NPZ file supplies `train_segs` and `test_segs`; this repository does not randomly split individual time bins. The pipeline takes the first `floor(0.2 * len(train_segs))` entries of the supplied training-segment array as validation segments and uses the remaining entries for parameter fitting. The supplied `test_segs` are used only for final decoding.

Consequently, the implemented split is **segment-index-level**, not a random time-bin split. In this dataset representation, each segment is a contiguous interval addressed by its corresponding `starts[s]:ends[s]` pair. The code keeps whole intervals together when assigning train, validation, and test segment IDs. It does not independently verify from raw session/trial metadata that every segment ID corresponds to a unique biological trial or that no trial was divided upstream when the NPZ was created. Therefore, trial-level independence is only as reliable as the provenance of the supplied `train_segs` and `test_segs`; the repository code alone cannot certify the upstream split. No claim of session-level generalization is made.

The validation subset is selected by array order, not by a seeded random permutation. Reproducing the same validation assignment requires the same ordering of `train_segs` in the NPZ file.

## R² definition

For each velocity component (j\in\{x,y\}), predictions and targets from all decoded test segments are concatenated. The code computes the coefficient of determination over that pooled set:

$$
R^2_j = 1 -
\frac{\sum_{i=1}^{N}(y_{ij}-\hat y_{ij})^2}
{\sum_{i=1}^{N}(y_{ij}-\bar y_j)^2},
\qquad
\bar y_j = \frac{1}{N}\sum_{i=1}^{N}y_{ij}.
$$

The reported mean is the unweighted arithmetic average of the two axis-specific values:

$$
R^2_{\mathrm{mean}} = \frac{R^2_x+R^2_y}{2}.
$$

This is not the average of per-trial R² values. It is pooled across the concatenated test samples separately for each axis. A constant target component would make its denominator zero and the score undefined.

## Repository contents

- `run_benchmark.py`: executes the implemented Kalman evaluation and writes `results.csv`.
- `src/data/classic_models.py`: Kalman decoder and R² function.
- `src/data/data_pipeline.py`: dataset loading and feature/tensor preparation.
- `src/data/deep_models.py`: GRU model definitions.
- `src/data/mc_maze_20ms.npz`: preprocessed dataset used by the runner.
- `pred_vs_true.png`: example prediction plot.

## Reproduce the implemented Kalman result

The preprocessed dataset is committed in this repository, so no separate dataset download is required for this exact run.

~~~bash
git clone https://github.com/shabab-ibg/kalman-neural-decoder.git
cd kalman-neural-decoder
python -m pip install -r requirements.txt
python run_benchmark.py
~~~

The command writes `results.csv` in the repository root. That file is a generated output and is not committed by default. The command recomputes only the Kalman row. It does not train or evaluate the Wiener, hybrid, or GRU models.

For the original MC_Maze dataset and benchmark documentation, see the [Neural Latents Benchmark](https://neurallatents.github.io/) and the [NLB tools repository](https://github.com/neurallatents/nlb_tools). The committed NPZ is the direct input expected by this project; obtaining the upstream raw data is not part of the one-command reproduction path above.

## Limitations

- The current runner does not reproduce the historical Wiener, hybrid, or GRU metrics.
- The saved GRU mean ± standard deviation values cannot be independently checked without per-seed outputs or training scripts.
- The current Kalman implementation is a simple linear state-space decoder with a lagged observation. It should not be interpreted as an optimized or universally representative Kalman baseline.
- The reported simulation check (approximately R² = 0.55) has not been independently validated against a fully specified synthetic data generator, matched initialization, burn-in, and noise covariance settings.
- The average-position baseline is not a valid comparator for a velocity-prediction target.

## References

Pei et al., [Neural Latents Benchmark '21: Evaluating latent variable models of neural population activity](https://arxiv.org/abs/2109.04463), NeurIPS 2021.

![Predicted versus true hand velocity](pred_vs_true.png)
