import numpy as np


def compute_r2(y_true, y_pred):
    """Compute per-output R². Constant targets return NaN for that output."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residual = ((y_true - y_pred) ** 2).sum(axis=0)
    total = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - residual / total


class SingleLagKalmanDecoder:
    """Linear state-space decoder with a lagged neural observation model."""

    def __init__(self, lag_bins=6):
        self.k = lag_bins
        self.params = {}

    def fit(self, X, Y, starts, ends, train_segs):
        xp, xn, xs_, zs = [], [], [], []
        for s in train_segs:
            a, b = starts[s], ends[s]
            if b - a < 2:
                continue
            xp.append(Y[a:b - 1])
            xn.append(Y[a + 1:b])
            if b - a > self.k:
                xs_.append(Y[a + self.k:b])
                zs.append(X[a:b - self.k])

        if not xp or not xs_:
            raise ValueError("Insufficient training data for the configured Kalman lag.")

        xp, xn = np.concatenate(xp), np.concatenate(xn)
        xs_, zs = np.concatenate(xs_), np.concatenate(zs)
        mu = xs_.mean(axis=0)

        A = np.linalg.lstsq(xp - mu, xn - mu, rcond=None)[0].T
        process_residual = xn - mu - (xp - mu) @ A.T
        Q = np.atleast_2d(np.cov(process_residual.T))

        D = np.column_stack([xs_ - mu, np.ones(len(xs_))])
        coef = np.linalg.lstsq(D, zs, rcond=None)[0]
        C, d = coef[:2].T, coef[2]
        obs_residual = zs - D @ coef
        R = np.atleast_2d(np.cov(obs_residual.T)) + 1e-3 * np.eye(zs.shape[1])

        self.params = {
            "A": A,
            "Q": Q,
            "C": C,
            "d": d,
            "mu": mu,
            "CtRi": C.T @ np.linalg.solve(R, np.eye(R.shape[0])),
            "CtRiC": C.T @ np.linalg.solve(R, C),
            "P0": np.atleast_2d(np.cov(xs_.T)),
        }

    def decode_sequence(self, X, Y, starts, ends, segs, L0=9):
        if not self.params:
            raise RuntimeError("Call fit() before decode_sequence().")

        A, Q = self.params["A"], self.params["Q"]
        C, d, mu = self.params["C"], self.params["d"], self.params["mu"]
        CtRi, CtRiC = self.params["CtRi"], self.params["CtRiC"]
        preds, truth = [], []

        for s in segs:
            a, b = starts[s], ends[s]
            if b - a <= L0 or b - a <= self.k:
                continue

            x = np.zeros(2)
            P = self.params["P0"].copy()
            out = []
            for t in range(self.k, b - a):
                x = A @ x
                P = A @ P @ A.T + Q
                z = X[a + t - self.k] - d

                # Solve the information-form update without explicit inversion.
                prior_precision = np.linalg.pinv(P)
                posterior_cov = np.linalg.pinv(prior_precision + CtRiC)
                x = x + posterior_cov @ (CtRi @ (z - C @ x))
                P = posterior_cov
                out.append(x + mu)

            aligned = np.asarray(out)[L0 - self.k:]
            target = Y[a + L0:b]
            n = min(len(aligned), len(target))
            if n:
                preds.append(aligned[:n])
                truth.append(target[:n])

        if not preds:
            raise ValueError("No test predictions were generated; check segment lengths and L0.")

        return np.concatenate(truth, axis=0), np.concatenate(preds, axis=0)
