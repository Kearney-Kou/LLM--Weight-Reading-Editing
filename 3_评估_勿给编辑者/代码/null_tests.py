"""How to run: from the 3_评估_勿给编辑者 root, run python3 代码/null_tests.py
Null-hypothesis tests: 'If we change the same kind and the same number of weights at random, is there an equally large gain?'
(a) Null distribution for 2A: pick 9 random |dt| (1..40), 18 time offsets on both sides together, set W1 to 0 + b2 mean compensation. 20 times.
(b) Null distribution for 2A (ignoring structure): set a random 22.2% of the positions in W1 to 0 (same number of changes as 2A's edit 1) + b2 mean compensation. 10 times.
(c) Full ranking for 2B: remove each of the 28 feature dimensions in turn (set to 0 + b2 mean compensation), and see where dimension 26 ranks.
All results are changes relative to the original network on B. Compensation always uses the unlabelled h1 mean over 40000 random frames of A."""
import numpy as np, torch, sys
from wlib import Frames, predict, NAMES
torch.set_num_threads(2)
w0 = {k: v.astype(np.float32) for k, v in np.load('网络/subject_net.npz').items()}
A = Frames('数据/dev_A.npz'); B = Frames('数据/dev_B.npz')
idx = torch.from_numpy(np.sort(np.random.default_rng(0).choice(len(A), 40000, replace=False)))
XA = A.windows(idx).numpy()
base = (predict(w0, B) == B.y).float().mean().item(); print('原网络 B', round(base, 4), flush=True)
def h1mean(w): return np.maximum(XA @ w['W1'] + w['b1'], 0).mean(0)
m0 = h1mean(w0)
def zero_mask(mask):
    w = {k: v.copy() for k, v in w0.items()}; w['W1'][mask] = 0
    w['b2'] = w['b2'] + (m0 - h1mean(w)) @ w0['W2']
    return (predict(w, B) == B.y).float().mean().item() - base
part = sys.argv[1]
rng = np.random.default_rng(42)
if part == 'a':
    for i in range(20):
        ds = np.sort(rng.choice(np.arange(1, 41), 9, replace=False))
        ts = sorted(set([40 + d for d in ds] + [40 - d for d in ds]))
        m = np.zeros((81, 28, 1024), bool); m[ts] = True
        print(f'(a) {i:2d} |dt|={ds.tolist()}: {zero_mask(m.reshape(2268, 1024)):+.4f}', flush=True)
if part == 'b':
    for i in range(10):
        m = rng.random((2268, 1024)) < 516096 / (2268 * 1024)
        print(f'(b) {i:2d} 随机 22.2% 位置: {zero_mask(m):+.4f}', flush=True)
if part == 'c':
    for f in range(28):
        m = np.zeros((81, 28, 1024), bool); m[:, f] = True
        print(f'(c) f={f:2d}: {zero_mask(m.reshape(2268, 1024)):+.4f}', flush=True)
