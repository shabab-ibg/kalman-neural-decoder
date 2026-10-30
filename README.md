# Decoding Hand Velocity from Motor Cortex Spikes: Kalman, Wiener, and GRU Decoders

An investigation into neural decoding using intracortical spiking activity from the Neural Latents Benchmark (MC_Maze) dataset.

This project investigates linear regression, state-space estimation, and recurrent neural networks for predicting hand velocity from motor cortex spike trains. The repository currently does not contain an executable end-to-end benchmark for every model listed below. The current `run_benchmark.py` recomputes only the Kalman result; the other values are historical reported scores and are not reproduced by that script.

## 1. Previously Reported Results (Not Yet Reproduced End-to-End)

| Model Architecture | Input History | Test R² (Mean Across X and Y Velocity) |
|---|---|---:|
| Wiener Filter (Ridge Regression) | 200 ms | 0.535 |
| Kalman Filter (Single-Lag Observation Model) | 1 bin, with a 120 ms lag | 0.109 |
| Wiener–Kalman Hybrid | 200 ms | 0.545 |
| GRU (Windowed Input) | 200 ms | 0.888 ± 0.002 |
| **GRU (Full-Segment History)** | **Entire Segment** | **0.915 ± 0.002** |

**Reproducibility status:** The values in this table are historical reported scores, not a fresh, end-to-end comparison produced by the current benchmark runner. The runner currently recomputes only the Kalman decoder score and writes that result to `results.csv`. The Wiener, hybrid, and GRU scores are not recomputed by the runner. The GRU standard deviations (± 0.002) have not been independently verified from saved per-seed predictions or fresh runs. Treat these numbers as provisional until all models are executed on the same held-out trials with documented preprocessing, target alignment, and metric calculation.

### Key Findings

- **Temporal context matters:** The multi-lag Wiener filter substantially outperformed the single-lag Kalman configuration.
- **Recurrent decoding performed best:** Both GRU configurations achieved higher test R² than the linear baselines in the reported experiments.
- **Longer temporal context improved performance:** The full-segment GRU achieved the highest reported R², although this comparison does not isolate sequence length from other architectural or training differences.
- **Validation controls:** Additional tests examined future-information leakage, cross-trial alignment, and the predictive value of simple temporal or spatial baselines.

These comparisons are provisional because the repository's current runner does not reproduce all listed models. They should not be interpreted as a verified ranking of neural decoding architectures until the full evaluation is executable and controlled.

---

## 2. Dataset and Objective

**Dataset:** Neural Latents Benchmark, MC_Maze.

**Task:** Predict hand velocity from intracortical motor cortex spike trains.

The decoding problem can be expressed as:

\[
\hat{\mathbf{v}}_t = f(\mathbf{s}_{1:t})
\]

where:

- \(\mathbf{s}_{1:t}\) represents the neural observations available up to time \(t\).
- \(\mathbf{v}_t = [v_x, v_y]\) represents the target hand-velocity vector.
- \(f\) is the decoding model.
- \(\hat{\mathbf{v}}_t\) is the predicted velocity.

The central experimental question is how different model families use neural activity and temporal context to reconstruct movement.

## 3. Model Architectures

### 3.1. Multi-Lag Wiener Filter

The linear baseline uses ridge regression on a history window of neural observations.

For a history of \(L\) bins:

\[
\hat{\mathbf{v}}_t =
\mathbf{W}^{\top}
\begin{bmatrix}
\mathbf{s}_{t-L+1}\\
\vdots\\
\mathbf{s}_t
\end{bmatrix}
+\mathbf{b}
\]

Here, \(\mathbf{W}\) contains the learned regression coefficients and \(\mathbf{b}\) is the intercept.

The reported configuration uses a 200 ms history window.

This model provides a linear baseline for measuring the predictive value of temporally extended neural activity.

### 3.2. State-Space Kalman Filter

The Kalman decoder uses a linear state-space formulation:

\[
\mathbf{x}_t = A\mathbf{x}_{t-1}+\mathbf{w}_t
\]

\[
\mathbf{z}_t = C\mathbf{x}_{t-k}+\mathbf{d}+\mathbf{v}_t
\]

where:

- \(\mathbf{x}_t\) is the latent kinematic state.
- \(\mathbf{z}_t\) is the neural observation vector.
- \(A\) describes state transitions.
- \(C\) maps the latent state to neural observations.
- \(\mathbf{w}_t\) and \(\mathbf{v}_t\) represent process and observation noise.
- \(k\) denotes the observation lag used in the implemented model.

Under the standard formulation, the noise terms are assumed to be Gaussian, with covariance matrices \(Q\) and \(R\), respectively.

The evaluated configuration uses a single-lag observation model. Its performance is compared against models that explicitly incorporate a longer history of neural activity.

**Implementation note:** The exact state variables, lag convention, and observation timing should be documented alongside the equations to make the formulation reproducible.

### 3.3. Gated Recurrent Unit (GRU)

The GRU decoder models temporal dependencies in neural activity using gated recurrent units.

Two configurations were evaluated:

- **Windowed GRU:** Uses a 200 ms history of neural observations.
- **Full-segment GRU:** Processes the available sequence across an entire segment.

The windowed model restricts the temporal context presented to the network. The full-segment model can exploit a longer sequence, potentially capturing temporal relationships that are unavailable to short-window models.

The full-segment configuration achieved the highest reported test R².

The comparison evaluates these specific implementations; additional controlled experiments would be required to attribute the performance difference exclusively to sequence length.

---

## 4. Comparative Analysis

The single-lag Kalman filter achieved a test R² of 0.109, compared with 0.535 for the multi-lag Wiener filter.

The Wiener–Kalman hybrid achieved 0.545, while the two GRU configurations reached 0.888 and 0.915.

These results suggest that the temporal representation and model configuration are important factors in this decoding task.

### Why did the single-lag Kalman configuration underperform?

Several explanations merit investigation:

**1. Restricted temporal context**

The Wiener filter incorporates a 200 ms history window, whereas the evaluated Kalman configuration uses a single-lag observation model. This difference may limit the information available to the Kalman decoder.

**2. Neural population structure**

Intracortical spike trains can be sparse and heterogeneous. The relationship between population activity and movement may not be adequately represented by the particular linear observation model used here.

**3. Model specification**

The choice of latent state, observation mapping, lag alignment, and noise covariance can affect Kalman decoding performance. These factors should be examined before attributing the performance difference to inherent limitations of linear state-space models.

The current results establish a performance difference between the evaluated configurations. They do not independently establish which of these mechanisms explains that difference.

---

## 5. Validation and Sanity Controls

Additional controls were implemented to investigate whether the reported decoding performance depended on unintended information leakage or trial-specific alignment.

### 5.1. Future-Information Leakage Test

Future tracking inputs were replaced with uniform random noise, and the resulting predictions for past velocity coordinates were compared.

The reported change was zero.

This is evidence that the tested predictions were invariant to that particular perturbation. It does not independently prove that the entire training and evaluation pipeline is free of temporal leakage.

### 5.2. Cross-Trial Control

Movement blocks were shuffled across trials to test whether decoding performance depended on the alignment between neural activity and its corresponding movement targets.

The reported control produced an R² of approximately -1.0.

This indicates poor predictive performance under the tested shuffled condition. The precise interpretation depends on the shuffling procedure and the definition of the evaluation set.

### 5.3. Baseline Control

A simple average-position baseline was evaluated, producing a reported R² near zero.

This provides a reference for interpreting the predictive performance of the trained models.

Because this baseline predicts position while the principal task predicts velocity, the target definitions should be checked before treating the scores as directly comparable.

---

## 6. Simulation-Based Validation

The Kalman implementation was also evaluated on simulated trajectories generated under the intended Gaussian state-space assumptions.

The reported simulation result was approximately \(R^2 = 0.55\).

This experiment provides a useful implementation check under controlled conditions. It suggests that the model can recover meaningful structure in simulated data, but it does not establish that the model is optimally specified for real neural recordings.

The difference between simulated and real-data performance motivates further investigation of observation-model design, temporal context, and neural-to-kinematic relationships.

---

## 7. Reproducibility and Evaluation

The reported evaluation includes comparisons across multiple model configurations and three random initializations for the GRU experiments.

The repository's results artifacts include per-axis evaluation outputs in `results.csv`.

For a reproducible comparison, the evaluation protocol should specify:

- Train, validation, and test split construction.
- Whether splitting occurs at the trial or time-bin level.
- Neural activity preprocessing and normalization.
- Velocity target construction and temporal alignment.
- Model hyperparameters and regularization.
- The precise definition of the reported R² metric.
- Whether all models use identical held-out test observations.
- Whether the reported GRU variability reflects random initialization alone or any additional sources of variation.

These details are essential for interpreting differences between model architectures.

## 8. Conclusion

This project compares linear filters, a state-space Kalman decoder, and recurrent neural networks for predicting hand velocity from motor cortex activity.

In the reported experiments, the full-segment GRU achieved the highest test R², followed by the windowed GRU. Both outperformed the evaluated linear baselines.

The results highlight the importance of temporal context and model specification in neural decoding. The contrast between simulated and real-data performance also motivates further investigation into how linear state-space models can be adapted to the statistical structure of biological signals.

The broader objective is to understand not only which decoder performs best, but also which assumptions and temporal representations explain its performance.
