"""运行方式：在 3_评估_勿给编辑者 根目录下执行 python3 代码/null_tests.py
零假设检验：'随便改同一类、同样多的权重，也会有这么大的提升吗？'
(a) 2A 的零分布：随机挑 9 个 |dt|（1..40），两侧共 18 个时间偏移，W1 置 0 + b2 均值补偿。20 次。
(b) 2A 的零分布（不看结构）：在 W1 里随机挑 22.2% 的位置置 0（和 2A 的修改 1 改动个数相同）+ b2 均值补偿。10 次。
(c) 2B 的完整排名：把 28 个特征维逐一去掉（置 0 + b2 均值补偿），看第 26 维排第几。
所有结果都是 B 上相对原网络的变化。补偿统一用 A 上随机 40000 帧的无标签 h1 均值。"""
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
