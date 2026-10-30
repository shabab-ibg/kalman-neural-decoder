# Decoding Hand Velocity from Motor Cortex Spikes: Kalman, Wiener, and GRU Decoders

Offline decoding of hand velocity from intracortical spiking activity (Neural Latents Benchmark, MC_Maze). This repository features a comprehensive comparison of linear filters, state-space estimators, and recurrent neural networks (GRUs), alongside an analytical breakdown of linear-Gaussian limits on sparse physiological signals.

## ?? Summary Performance Metrics

| Model Architecture | Input History Window | Test $R2$ (Mean of $v_x, v_y$) |
| :--- | :---: | :---: |
| **Wiener Filter** (Ridge regression on lagged spikes) | 200 ms | 0.535 |
| **Kalman Filter** (Linear observation model, single lag) | 1 bin (120 ms lag) | 0.109 |
| **Wiener Output + Kalman Smoother Hybrid** | 200 ms | 0.545 |
| **GRU (Windowed Input)** | 200 ms | $0.888 \pm 0.002$ (3 seeds) |
| **GRU (Full Segment History)** | Whole Segment | **$0.915 \pm 0.002$** (3 seeds) |

*$\pm$ indicates standard deviation boundaries across 3 distinct random initializations. Detailed breakdowns per operational control coordinate axis are exported natively into `results.csv` icons.*

---

## ?? Mathematical Architectures

### 1. Causal Wiener History Filter
The baseline is structured via an Ordinary Least Squares (OLS) multi-lag regularized Ridge configuration. Inputs map an integrated chronological context window ($L = 10$ bins $\rightarrow$ 200 ms) of neuronal activations directly to physical output targets:
$$\hat{y}_t = X_{t-L:t} \cdot W + b$$

### 2. Standard State-Space Kalman Estimator
Structured explicitly in tracking spaces using continuous Gaussian mechanics:
*   **System Latent Dynamics:** $x_t = A x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q)$
*   **System Sensory Matrix emissions:** $z_t = C x_{t-k} + d + v_t, \quad v_t \sim \mathcal{N}(0, R)$

### 3. Non-Linear Neural Sequences (GRU)
Deep Recurrent Networks running under square-root input scaling paths to minimize transient firing spikes. Run in local history tracking frames (200 ms windows) or trial-long continuous operations.

---

## ?? Diagnosis: Why the Plain Kalman Decoder Underperforms

Our tracking validation verified that the filter formulation code behaves correctly under ideal conditions. When executed over simulated trajectories containing standard Gaussian variances, the custom system maps accurately ($R2 \approx 0.55$). 

The massive performance drop on real neural data stems from standard linear modeling assumptions failing when applied to raw biological signals:
1. **Information Bottleneck:** Limiting regression dependencies to an isolated, singular historical bin drops structural capability dramatically ($R2 \le 0.165$).
2. **Sparsity & Non-Linear Tuning:** Real cortical arrays feature highly sparse emission traces ($\sim 95\%$ blank bins). Because single-unit response fields map non-linearly to arm movement profiles, linear state update vectors struggle to maintain track.

---

## ??? Sanity Controls on the GRU Model
To rule out data leakage or overfitting, three explicit validation gates were implemented:
*   **Causality Check:** Swapping future tracking bins with uniform random noise yields a delta shift of exactly $0.0$ on past velocity coordinates.
*   **Cross-Trial Controls:** Correlating a trial's tracking parameters against an entirely shuffled movement block drops decoding target capture levels to an un-entangled state ($R2 \approx -1.0$).
*   **Temporal Templates:** Substituting neural dynamics with an average spatial position coordinate baseline nets a predictive evaluation threshold of $\sim 0.0$.
