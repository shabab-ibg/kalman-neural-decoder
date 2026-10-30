import numpy as np
import torch


class NLBMazePipeline:
    """Load preprocessed MC_Maze neural activity and kinematic targets."""

    def __init__(self, file_path="data/mc_maze_20ms.npz", history_len=10, device="cpu"):
        payload = np.load(file_path)
        self.X = payload["X"]
        self.Y = payload["Y"]
        self.starts = payload["starts"]
        self.ends = payload["ends"]
        self.train_segs = payload["train_segs"]
        self.test_segs = payload["test_segs"]
        self.L = history_len
        self.device = device

        # Reserve the first 20% of training segments for validation.
        n_val = len(self.train_segs) // 5
        self.val_segs = self.train_segs[:n_val]
        self.tr_s = self.train_segs[n_val:]
        if len(self.tr_s) == 0:
            raise ValueError("No training segments remain after the train/validation split.")

        # Fit target normalization on training segments only.
        trY = np.concatenate(
            [self.Y[self.starts[s]:self.ends[s]] for s in self.tr_s], axis=0
        )
        self.ymu = trY.mean(axis=0)
        self.ysd = trY.std(axis=0)
        self.ysd = np.where(self.ysd == 0, 1.0, self.ysd)

    def generate_wiener_features(self, segments):
        Xs, Ys = [], []
        for s in segments:
            a, b = self.starts[s], self.ends[s]
            if b - a < self.L:
                continue
            xs = self.X[a:b]
            windows = np.stack(
                [xs[k:len(xs) - self.L + 1 + k] for k in range(self.L)],
                axis=1,
            )
            Xs.append(windows.reshape(len(windows), -1))
            Ys.append(self.Y[a + self.L - 1:b])
        if not Xs:
            raise ValueError("No segments are long enough to generate Wiener features.")
        return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)

    def pack_sequential_tensors(self, segments):
        valid = [s for s in segments if self.ends[s] - self.starts[s] >= self.L]
        if not valid:
            raise ValueError("No segments are long enough to pack sequential tensors.")

        max_t = max(self.ends[s] - self.starts[s] for s in valid)
        xb = np.zeros((len(valid), max_t, self.X.shape[1]), dtype=np.float32)
        yb = np.zeros((len(valid), max_t, 2), dtype=np.float32)
        mask = np.zeros((len(valid), max_t), dtype=np.float32)

        for i, s in enumerate(valid):
            a, b = self.starts[s], self.ends[s]
            n = b - a
            xb[i, :n] = np.sqrt(self.X[a:b])
            yb[i, :n] = (self.Y[a:b] - self.ymu) / self.ysd
            mask[i, self.L - 1:n] = 1.0

        return tuple(
            torch.tensor(v, dtype=torch.float32, device=self.device)
            for v in (xb, yb, mask)
        )

    def generate_windowed_tensors(self, segments):
        xs, ys = [], []
        for s in segments:
            a, b = self.starts[s], self.ends[s]
            if b - a < self.L:
                continue
            xx = np.sqrt(self.X[a:b])
            windows = np.stack(
                [xx[k:len(xx) - self.L + 1 + k] for k in range(self.L)],
                axis=1,
            )
            xs.append(windows)
            ys.append((self.Y[a + self.L - 1:b] - self.ymu) / self.ysd)

        if not xs:
            raise ValueError("No segments are long enough to generate windowed tensors.")

        return (
            torch.tensor(np.concatenate(xs, axis=0), dtype=torch.float32, device=self.device),
            torch.tensor(np.concatenate(ys, axis=0), dtype=torch.float32, device=self.device),
        )
