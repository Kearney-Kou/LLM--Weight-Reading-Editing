"""运行方式：在 3_评估_勿给编辑者 根目录下执行 python3 代码/targeted_controls.py
更公平的对照：同一种修改，换一个目标。
(1) 2B 类：把某一维特征 f 在全部 81 帧上的 W1 行置 0 + b2 均值补偿（和 2B 的做法完全一样），f 取 26（2B 选的）和其他 8 个随机维度。
(2) 2A 类：只做 2A 的修改 1+2（把一段时间偏移的 W1 置 0 + b2 均值补偿），比较 2A 选的 |dt|=30..38 和更近的一段 |dt|=20..28、以及最远的 |dt|=32..40。
补偿统一用 A 上随机 40000 帧的无标签 h1 均值（与 2B 的做法一致）。"""
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
