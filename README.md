# Kalman Neural Decoder

A comparative study of neural decoding methods for predicting two-dimensional hand velocity from intracortical motor-cortex spike activity in the Neural Latents Benchmark (MC_Maze) dataset.

The project evaluates linear regression, a Kalman state-space decoder, a Wiener–Kalman hybrid, and recurrent neural networks. The central question is how model structure and temporal context affect kinematic decoding.

> **Scope:** This is an offline analysis of an existing intracortical neural dataset. It is not a real-time brain–computer interface, a wearable EEG system, or a clinical device.

## Results

The table reports the test-set mean R² across the X- and Y-velocity targets, as recorded in the project experiments.

| Decoder | Temporal input | Test R² |
|---|---|---:|
| Wiener filter (ridge regression) | 200 ms history | 0.535 |
| Kalman filter (single-lag observation model) | One observation, 120 ms lag | 0.109 |
| Wiener–Kalman hybrid | 200 ms history | 0.545 |
| GRU | 200 ms window | 0.888 ± 0.002 |
| **GRU** | **Full segment** | **0.915 ± 0.002** |

The GRU values are reported as mean ± standard deviation across three random initializations. The other values are the reported experiment scores; no across-seed uncertainty is implied.

### What the results suggest

- The tested multi-lag Wiener filter outperformed the tested single-lag Kalman configuration.
- The Wiener–Kalman hybrid slightly exceeded the Wiener baseline in this evaluation.
- Both GRU configurations achieved higher test R² than the evaluated linear configurations.
- The full-segment GRU achieved the highest reported score.

These are comparisons between the **specific implementations and evaluation setup in this repository**. They do not establish that GRUs are universally superior to Kalman filters, or that temporal context alone explains the GRU performance. The full-segment and windowed GRUs differ in more than sequence length unless all other architectural and training choices are held constant.

## Dataset and task

- **Dataset:** Neural Latents Benchmark — MC_Maze.
- **Input:** Intracortical motor-cortex spike activity.
- **Target:** Two-dimensional hand velocity, represented as \(\mathbf{v}_t = [v_x, v_y]\).
- **Task:** Estimate hand velocity from neural observations.

In general form, the decoder estimates

\[
\hat{\mathbf{v}}_t = f(\mathbf{s}_{1:t}),
\]

where \(\mathbf{s}_{1:t}\) denotes the neural observations available to the decoder and \(f\) is the learned or specified decoding function.

The task is kinematic prediction from recorded neural activity. It should not be conflated with decoding arbitrary thoughts, imagined speech, or music.

## Decoders

### 1. Wiener filter

The Wiener baseline is implemented as ridge regression over a history window of neural observations:

\[
\hat{\mathbf{v}}_t =
W^\top
\begin{bmatrix}
\mathbf{s}_{t-L+1} \\
\vdots \\
\mathbf{s}_t
\end{bmatrix}
+ \mathbf{b}.
\]

The reported configuration uses a 200 ms history window. This provides a linear baseline that can use multiple recent observations rather than relying on a single time bin.

### 2. Kalman filter

The Kalman decoder uses a linear state-space model. In generic form:

\[
\mathbf{x}_t = A\mathbf{x}_{t-1} + \mathbf{w}_t,
\]

\[
\mathbf{z}_t = C\mathbf{x}_{t-k} + \mathbf{d} + \mathbf{v}_t,
\]

where \(\mathbf{x}_t\) is the kinematic state, \(\mathbf{z}_t\) is the neural observation, \(A\) is the state-transition matrix, \(C\) is the observation matrix, and \(k\) denotes the implemented observation lag. The noise terms are commonly modelled with covariance matrices \(Q\) and \(R\).

The evaluated configuration uses a single-lag observation model with a reported 120 ms lag. The exact state definition, lag convention, fitting procedure and covariance estimation should be taken from the implementation when reproducing the experiment.

### 3. Wiener–Kalman hybrid

The hybrid combines the project's linear temporal-regression and state-space components. It achieved a reported test R² of 0.545. The exact construction and fitting procedure should be inspected in the implementation before interpreting it as a general hybrid architecture.

### 4. GRU

A gated recurrent unit (GRU) network models temporal dependencies in the neural observations.

Two configurations were evaluated:

- **Windowed GRU:** receives a 200 ms input window.
- **Full-segment GRU:** processes the available sequence across a full segment.

The full-segment configuration achieved the highest reported test R². This is an empirical result for the evaluated setup, not proof that longer input sequences will always improve neural decoding.

## Validation and controls

The experiments include additional checks intended to probe alignment and unintended predictive information.

### Future-information perturbation

Future tracking inputs were replaced with uniform random noise, and predictions for past velocity coordinates were compared. The reported change was zero.

This establishes invariance to that particular perturbation. It does **not**, by itself, prove that every stage of preprocessing, training and evaluation is free of temporal leakage.

### Cross-trial control

Movement blocks were shuffled across trials. The reported R² was approximately -1.0, indicating poor predictive performance under the tested shuffled condition. Interpretation depends on the precise shuffle procedure and evaluation split.

### Average-position baseline

A simple average-position baseline reportedly produced an R² near zero. Because this baseline predicts position while the main task predicts velocity, its score should not be treated as a directly comparable decoder result unless the target definitions and evaluation procedure are made identical.

### Simulation check

The Kalman implementation was also tested on simulated trajectories generated under the intended Gaussian state-space assumptions. The reported simulation result was approximately \(R^2 = 0.55\).

This is an implementation sanity check under controlled assumptions. It does not establish that the model is correctly specified or optimally tuned for real neural recordings.

## Evaluation considerations

The reported scores should be interpreted alongside the evaluation protocol. A reproducible comparison should document:

- How train, validation and test sets are split, including whether splitting is by trial or time bin.
- Neural preprocessing, binning, normalization and any smoothing.
- The construction and temporal alignment of velocity targets.
- The exact state definition and observation lag used by the Kalman model.
- Model hyperparameters, regularization and training procedures.
- Whether every model is evaluated on identical held-out observations.
- The definition of R² and whether it is computed separately per axis before averaging.
- Which sources of variability are represented by the reported GRU standard deviations.

A high score is meaningful only if the test set is held out appropriately and the same evaluation rules are applied across models.

## Limitations

- The results concern one dataset and one decoding task.
- Intracortical spike recordings are not equivalent to non-invasive scalp EEG.
- The comparison covers particular model configurations, not every possible Kalman, Wiener or recurrent decoder.
- The reported controls address specific failure modes; they are not a complete audit of the pipeline.
- The README's summary metrics should be checked against the underlying result files and scripts before being treated as independently reproduced.

## Reproducibility

The experiment summary above records the reported configurations and scores. For exact reproduction, consult the source code and dataset preparation steps in the repository. Add the precise environment, dependency versions, dataset access instructions, preprocessing configuration, training commands and evaluation commands here as those details are verified.

## Conclusion

In these experiments, the full-segment GRU achieved the highest reported test R² (0.915 ± 0.002), while the single-lag Kalman configuration scored substantially lower (0.109) than the multi-lag Wiener baseline (0.535).

The defensible conclusion is not that Kalman filtering is inherently inferior. The evaluated Kalman configuration used a restricted observation history, and its state-space specification may not have captured the temporal structure available to the other models. Further controlled comparisons are needed to isolate the causes of the performance gap.

The project is an investigation into how temporal representations and model assumptions affect neural-to-kinematic decoding.
