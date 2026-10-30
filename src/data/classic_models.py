import numpy as np

def compute_r2(y_true, y_pred):
    return 1 - ((y_true - y_pred) ** 2).sum(0) / ((y_true - y_true.mean(0)) ** 2).sum(0)

class SingleLagKalmanDecoder:
    def __init__(self, lag_bins=6):
        self.k = lag_bins
        self.params = {}

    def fit(self, X, Y, starts, ends, train_segs):
        xp, xn, xs_, zs = [], [], [], []
        for s in train_segs:
            a, b = starts[s], ends[s]
            xp.append(Y[a:b-1])
            xn.append(Y[a+1:b])
            if b - a > self.k:
                xs_.append(Y[a+self.k:b])
                zs.append(X[a:b-self.k])
        xp, xn = np.concatenate(xp), np.concatenate(xn)
        xs_, zs = np.concatenate(xs_), np.concatenate(zs)
        mu = xs_.mean(0)

        A = np.linalg.lstsq(xp - mu, xn - mu, rcond=None)[0].T
        Q = np.cov((xn - mu - (xp - mu) @ A.T).T)
        D = np.c_[xs_ - mu, np.ones(len(xs_))]
        coef = np.linalg.lstsq(D, zs, rcond=None)[0]
        C, d = coef[:2].T, coef[2]
        R = np.cov((zs - D @ coef).T) + 1e-3 * np.eye(zs.shape[1])

        self.params = dict(A=A, Q=Q, C=C, d=d, mu=mu, CtRi=C.T @ np.linalg.inv(R),
                           CtRiC=C.T @ np.linalg.inv(R) @ C, P0=np.cov(xs_.T))

    def decode_sequence(self, X, Y, starts, ends, segs, L0=9):
        A, Q, C, d, mu = self.params["A"], self.params["Q"], self.params["C"], self.params["d"], self.params["mu"]
        CtRi, CtRiC = self.params["CtRi"], self.params["CtRiC"]
        preds, truth = [], []
        for s in segs:
            a, b = starts[s], ends[s]
            if b - a <= L0: continue
            x, P, out = np.zeros(2), self.params["P0"].copy(), []
            for t in range(self.k, b - a):
                x = A @ x
                P = A @ P @ A.T + Q
                z = X[a + t - self.k] - d
                P = np.linalg.inv(np.linalg.inv(P) + CtRiC)
                x = x + P @ (CtRi @ (z - C @ x))
                out.append(x + mu)
            preds.append(np.array(out)[L0 - self.k:])
            truth.append(Y[a + L0:b])
        return np.concatenate(truth), np.concatenate(preds)
