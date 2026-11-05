"""How to run: from the 3_评估_勿给编辑者 root, run python3 代码/targeted_controls.py
Fairer controls: the same kind of edit, aimed at a different target.
(1) 2B type: set the W1 rows of one feature dimension f over all 81 frames to 0 + b2 mean compensation (exactly as 2B did), with f = 26 (2B's choice) and 8 other random dimensions.
(2) 2A type: only 2A's edits 1+2 (set W1 for a range of time offsets to 0 + b2 mean compensation), comparing 2A's choice |dt|=30..38 with a nearer range |dt|=20..28 and the farthest range |dt|=32..40.
Compensation always uses the unlabelled h1 mean over 40000 random frames of A (consistent with what 2B did)."""
import numpy as np, torch
from wlib import Frames, predict, NAMES
torch.set_num_threads(2)
w0 = {k: v.astype(np.float32) for k, v in np.load('网络/subject_net.npz').items()}
A = Frames('数据/dev_A.npz'); B = Frames('数据/dev_B.npz')
idx = torch.from_numpy(np.sort(np.random.default_rng(0).choice(len(A), 40000, replace=False)))
XA = A.windows(idx).numpy()
base = (predict(w0, B) == B.y).float().mean().item(); print('原网络 B', round(base, 4), flush=True)
def h1mean(w): return np.maximum(XA @ w['W1'] + w['b1'], 0).mean(0)
m0 = h1mean(w0)
def zero_rows(rows):
    w = {k: v.copy() for k, v in w0.items()}; w['W1'][rows] = 0
    w['b2'] = w['b2'] + (m0 - h1mean(w)) @ w0['W2']
    return (predict(w, B) == B.y).float().mean().item() - base
rng = np.random.default_rng(3)
fs = [26] + sorted(rng.choice([f for f in range(28) if f != 26], 8, replace=False).tolist())
for f in fs:
    print(f'(1) 去掉特征维 f={f}: B 上 {zero_rows(np.arange(81) * 28 + f):+.4f}', flush=True)
for lo, hi in [(30, 38), (20, 28), (32, 40)]:
    ts = [40 + d for d in range(-40, 41) if lo <= abs(d) <= hi]
    rows = (np.array(ts)[:, None] * 28 + np.arange(28)).ravel()
    print(f'(2) 去掉 |dt|={lo}..{hi}（{len(ts)} 个偏移）: B 上 {zero_rows(rows):+.4f}', flush=True)
