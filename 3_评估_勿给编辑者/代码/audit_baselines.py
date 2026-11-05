"""运行方式：在 3_评估_勿给编辑者 根目录下执行 python3 代码/audit_baselines.py
审查补充实验（B 上相对原网络的变化）：
(i)   2A 的修改 1+2 精确版：取 2A 交上来的网络，把 W3、b3 恢复成原样（撤销修改 3–5）。
(ii)  量级剪枝基线：把 W1 里绝对值最小的 516,096 个权重置 0（与 2A 改动个数相同）+ b2 均值补偿。不需要 LLM。
(iii) 行范数基线：把"平均行范数最小"的 18 个时间偏移置 0 + 补偿。不需要 LLM，也不用标签。
(iv)  最小 η² 基线：在 A 上算每个特征维（中心帧）的类别区分度 η²，去掉 η² 最小的那一维 + 补偿。
(v)   探索：2A 修改 1+2 精确版 再叠加 去掉 f26 + 补偿。
补偿统一用 A 上随机 40000 帧的无标签 h1 均值（与 targeted_controls / null_tests 一致）。"""
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
