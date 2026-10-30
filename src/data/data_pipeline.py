import numpy as np
import torch

class NLBMazePipeline:
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
        
        # Exact 80/20 train/validation slice from user notebook
        n_val = len(self.train_segs) // 5
        self.val_segs = self.train_segs[:n_val]
        self.tr_s = self.train_segs[n_val:]
        
        # Scaling stats extracted from training velocities only
        trY = np.concatenate([self.Y[self.starts[s]:self.ends[s]] for s in self.tr_s])
        self.ymu, self.ysd = trY.mean(0), trY.std(0)

    def generate_wiener_features(self, segments):
        Xs, Ys = [], []
        for s in segments:
            a, b = self.starts[s], self.ends[s]
            if b - a < self.L: continue
            xs = self.X[a:b]
            windows = np.stack([xs[k : len(xs) - self.L + 1 + k] for k in range(self.L)], axis=1)
            Xs.append(windows.reshape(len(windows), -1))
            Ys.append(self.Y[a + self.L - 1 : b])
        return np.concatenate(Xs), np.concatenate(Ys)

    def pack_sequential_tensors(self, segments):
        valid = [s for s in segments if self.ends[s] - self.starts[s] >= self.L]
        max_t = max(self.ends[s] - self.starts[s] for s in valid)
        xb = np.zeros((len(valid), max_t, self.X.shape[1]), np.float32)
        yb = np.zeros((len(valid), max_t, 2), np.float32)
        mask = np.zeros((len(valid), max_t), np.float32)
        
        for i, s in enumerate(valid):
            a, b = self.starts[s], self.ends[s]
            n = b - a
            xb[i, :n] = np.sqrt(self.X[a:b])  # Sqrt transform used for GRU
            yb[i, :n] = (self.Y[a:b] - self.ymu) / self.ysd
            mask[i, (self.L - 1):n] = 1.0     # Score from bin 9 onward
        return tuple(torch.tensor(v).to(self.device) for v in (xb, yb, mask))

    def generate_windowed_tensors(self, segments):
        xs, ys = [], []
        for s in segments:
            a, b = self.starts[s], self.ends[s]
            if b - a < self.L: continue
            xx = np.sqrt(self.X[a:b])
            w = np.stack([xx[k:len(xx) - self.L + 1 + k] for k in range(self.L)], axis=1)
            xs.append(w)
            ys.append((self.Y[a + self.L - 1:b] - self.ymu) / self.ysd)
        return (torch.tensor(np.concatenate(xs), dtype=torch.float32).to(self.device),
                torch.tensor(np.concatenate(ys), dtype=torch.float32).to(self.device))
