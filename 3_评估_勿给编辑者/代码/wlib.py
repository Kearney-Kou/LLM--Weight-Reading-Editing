"""共用工具：数据加载、上下文窗口、网络前向（numpy 权重 -> torch）。
网络约定：x(2268) -> h1 = relu(x @ W1 + b1) (1024) -> h2 = relu(h1 @ W2 + b2) (1024) -> logits = h2 @ W3 + b3 (42)
W 的形状是 (输入, 输出)。输入 2268 = 81 帧 x 28 维，按"时间在外、特征在内"展平：下标 = t*28 + f，t=0..80，第 40 帧是被分类的中心帧。
"""
import numpy as np, torch
K = 40
NAMES = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']

class Frames:
    def __init__(self, path):
        d = np.load(path)
        self.x = torch.from_numpy(d['x'].astype(np.float32))
        self.y = torch.from_numpy(d['y'].astype(np.int64))
        self.lens = d['lens']
        # 每句前后各补 K 个零帧（不跨句）
        L = self.lens
        padded_start = np.concatenate([[0], np.cumsum(L + 2 * K)[:-1]])
        frame_start = np.concatenate([[0], np.cumsum(L)[:-1]])
        P = torch.zeros(int((L + 2 * K).sum()), 28)
        center = np.empty(int(L.sum()), np.int64)
        for u in range(len(L)):
            P[padded_start[u] + K: padded_start[u] + K + L[u]] = self.x[frame_start[u]: frame_start[u] + L[u]]
            center[frame_start[u]: frame_start[u] + L[u]] = padded_start[u] + K + np.arange(L[u])
        self.P = P
        self.center = torch.from_numpy(center)
        self.utt = torch.from_numpy(np.repeat(np.arange(len(L)), L))
        self.offs = torch.arange(-K, K + 1)
    def __len__(self):
        return len(self.y)
    def windows(self, idx):
        idx = torch.as_tensor(idx)
        return self.P[self.center[idx][:, None] + self.offs].reshape(len(idx), -1)

def to_torch(w):
    return {k: torch.as_tensor(np.asarray(w[k], np.float32)) for k in NAMES}

@torch.no_grad()
def predict(w, data, bs=16384, return_hidden=False):
    t = to_torch(w)
    preds = []
    for i in range(0, len(data), bs):
        x = data.windows(torch.arange(i, min(i + bs, len(data))))
        h1 = torch.relu(x @ t['W1'] + t['b1'])
        h2 = torch.relu(h1 @ t['W2'] + t['b2'])
        preds.append((h2 @ t['W3'] + t['b3']).argmax(1))
    return torch.cat(preds)

def accuracy(w, data):
    return (predict(w, data) == data.y).float().mean().item()
