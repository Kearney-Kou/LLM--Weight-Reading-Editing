import numpy as np, torch
K = 40
class TrainFrames:   # 训练数据用 float16 存，省内存
    def __init__(self, paths):
        ds = [np.load(p) for p in paths]
        x = np.concatenate([d['x'] for d in ds]); self.y = torch.from_numpy(np.concatenate([d['y'] for d in ds]).astype(np.int64))
        L = np.concatenate([d['lens'] for d in ds])
        ps = np.concatenate([[0], np.cumsum(L + 2 * K)[:-1]]); fs = np.concatenate([[0], np.cumsum(L)[:-1]])
        P = np.zeros((int((L + 2 * K).sum()), 28), np.float16); center = np.empty(int(L.sum()), np.int64)
        for u in range(len(L)):
            P[ps[u] + K: ps[u] + K + L[u]] = x[fs[u]: fs[u] + L[u]]
            center[fs[u]: fs[u] + L[u]] = ps[u] + K + np.arange(L[u])
        del x
        self.P = torch.from_numpy(P); self.center = torch.from_numpy(center); self.offs = torch.arange(-K, K + 1)
    def __len__(self): return len(self.y)
    def windows(self, idx):
        return self.P[self.center[idx][:, None] + self.offs].reshape(len(idx), -1).float()

