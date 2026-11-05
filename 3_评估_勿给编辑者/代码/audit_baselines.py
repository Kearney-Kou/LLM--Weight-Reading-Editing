"""How to run: from the 3_评估_勿给编辑者 root, run python3 代码/audit_baselines.py
Additional baselines (change relative to the original network on B):
(i)   Exact version of 2A's edits 1+2: take the network 2A submitted and restore W3, b3 to the original (undo edits 3–5).
(ii)  Magnitude-pruning baseline: set the 516,096 smallest-magnitude weights in W1 to 0 (same number as 2A changed) + b2 mean compensation. No LLM needed.
(iii) Row-norm baseline: set the 18 time offsets with the "smallest mean row norm" to 0 + compensation. No LLM needed, and no labels used.
(iv)  Minimum-η² baseline: on A, compute the class separability η² of each feature dimension (centre frame), remove the one dimension with the smallest η² + compensation.
(v)   Exploration: exact version of 2A's edits 1+2, plus removing f26 + compensation.
Compensation always uses the unlabelled h1 mean over 40000 random frames of A (consistent with targeted_controls / null_tests)."""
import numpy as np, torch
from wlib import Frames, predict, NAMES
torch.set_num_threads(2)
w0 = {k: v.astype(np.float32) for k, v in np.load('网络/subject_net.npz').items()}
A = Frames('数据/dev_A.npz'); B = Frames('数据/dev_B.npz')
idx = torch.from_numpy(np.sort(np.random.default_rng(0).choice(len(A), 40000, replace=False)))
XA = A.windows(idx).numpy()
def acc(w): return (predict(w, B) == B.y).float().mean().item()
base = acc(w0); print('原网络 B', round(base, 4), flush=True)
def h1mean(w): return np.maximum(XA @ w['W1'] + w['b1'], 0).mean(0)
m0 = h1mean(w0)
def zero_and_comp(mask, start=None):
    w = {k: v.copy() for k, v in (start or w0).items()}; w['W1'][mask] = 0
    w['b2'] = w0['b2'] + (m0 - h1mean(w)) @ w0['W2']
    return w
# (i)
e = {k: v.astype(np.float32) for k, v in np.load('网络/edited_2A.npz').items()}
e12 = {k: v.copy() for k, v in e.items()}; e12['W3'] = w0['W3'].copy(); e12['b3'] = w0['b3'].copy()
print(f'(i)   2A 修改1+2 精确版: {acc(e12) - base:+.4f}', flush=True)
# (ii)
thr = np.sort(np.abs(w0['W1']).ravel())[516096 - 1]
m = np.abs(w0['W1']) <= thr; print('      剪枝个数', int(m.sum()))
print(f'(ii)  量级剪枝 22.2%: {acc(zero_and_comp(m)) - base:+.4f}', flush=True)
# (iii)
rn = np.linalg.norm(w0['W1'].reshape(81, 28, 1024), axis=2).mean(1)
ts = np.argsort(rn)[:18]; print('      行范数最小的 18 个偏移 dt =', sorted((ts - 40).tolist()))
mm = np.zeros((81, 28, 1024), bool); mm[ts] = True
print(f'(iii) 行范数最小 18 个偏移: {acc(zero_and_comp(mm.reshape(2268, 1024))) - base:+.4f}', flush=True)
# (iv)
xc = A.P[A.center].numpy(); y = A.y.numpy()
eta = []
for f in range(28):
    v = xc[:, f]; c = np.bincount(y, minlength=42); mu = np.bincount(y, v, 42) / np.maximum(c, 1)
    eta.append(float((c * (mu - v.mean()) ** 2).sum() / ((v - v.mean()) ** 2).sum()))
fmin = int(np.argmin(eta)); print('      η² 最小的维度', fmin, '（η²=%.5f），次小 %s' % (eta[fmin], np.argsort(eta)[1:4].tolist()))
mm = np.zeros((81, 28, 1024), bool); mm[:, fmin] = True
print(f'(iv)  去掉 η² 最小的维 f={fmin}: {acc(zero_and_comp(mm.reshape(2268, 1024))) - base:+.4f}', flush=True)
# (v)
mm = np.zeros((81, 28, 1024), bool); mm[:, 26] = True
w5 = {k: v.copy() for k, v in e12.items()}; w5['W1'][mm.reshape(2268, 1024)] = 0
w5['b2'] = e12['b2'] + (h1mean(e12) - h1mean(w5)) @ w0['W2']
print(f'(v)   2A 修改1+2 + 去掉 f26: {acc(w5) - base:+.4f}', flush=True)
