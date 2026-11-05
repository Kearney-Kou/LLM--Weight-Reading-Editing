"""(1) 第 26 维信息量的更多检验（在 A 上）：
    η² = Σ_c n_c (μ_c − μ)² / Σ_i (x_i − μ)²（中心帧，按类别均值的方差解释比例）；
    互信息 I(分箱特征; 标签)：32 个等频箱，用"打乱标签"重复 20 次估计偏差（零基线）；
    同时对 x² 分箱（捕捉方差差异）；时间滞后 1..5 的自相关；与其他 27 维在 −3..+3 帧滞后上的最大绝对相关。
(2) B 上主表的按说话人 bootstrap（整位说话人一起重抽，2000 次）。"""
import numpy as np
d = np.load('数据/dev_A.npz'); x = d['x'].astype(np.float32); y = d['y'].astype(int); L = d['lens']
def mi(v, y, bins=32):
    q = np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]); b = np.searchsorted(q, v)
    J = np.zeros((bins, 42)); np.add.at(J, (b, y), 1); P = J / J.sum(); px = P.sum(1, keepdims=True); py = P.sum(0, keepdims=True)
    nz = P > 0; return float((P[nz] * np.log2(P[nz] / (px @ py)[nz])).sum())
rng = np.random.default_rng(0)
print('feat  eta2      MI(x) bits  null(95%)   MI(x^2) bits')
for f in [26, 25, 23, 24, 22, 0, 27]:
    v = x[:, f]; mu = v.mean(); c = np.bincount(y, minlength=42); m = np.bincount(y, v, 42) / np.maximum(c, 1)
    eta = (c * (m - mu) ** 2).sum() / ((v - mu) ** 2).sum()
    null = [mi(v, rng.permutation(y)) for _ in range(20)]
    print(f'{f:4d}  {eta:.5f}  {mi(v, y):.5f}     {np.percentile(null, 95):.5f}     {mi(v * v, y):.5f}')
s = np.concatenate([[0], np.cumsum(L)])
def lagcorr(a, b, k):   # 句内，a[t] 与 b[t+k]
    num = den1 = den2 = 0.0
    for u in range(len(L)):
        A = a[s[u]:s[u+1]]; B = b[s[u]:s[u+1]]
        if k > 0: A, B = A[:-k], B[k:]
        elif k < 0: A, B = A[-k:], B[:k]
        num += (A * B).sum(); den1 += (A * A).sum(); den2 += (B * B).sum()
    return num / np.sqrt(den1 * den2)
print('f26 自相关 lag1..5:', [round(lagcorr(x[:, 26], x[:, 26], k), 4) for k in range(1, 6)])
mx = max(abs(lagcorr(x[:, 26], x[:, g], k)) for g in range(28) if g != 26 for k in range(-3, 4))
print('f26 与其他维在 −3..+3 帧滞后上的最大 |相关|:', round(float(mx), 4))
# (2) 说话人 bootstrap
B = np.load('数据/dev_B.npz'); spk = np.array([n.split('-')[0] for n in B['names']])
C = np.load('结果/per_utt_correct.npz'); lens = C['lens']
us = np.unique(spk); idx = {sp: np.where(spk == sp)[0] for sp in us}
rng = np.random.default_rng(0); S = []
for _ in range(2000):
    pick = rng.choice(us, len(us)); S.append(np.concatenate([idx[p] for p in pick]))
print(f'B 共 {len(us)} 位说话人；按说话人 bootstrap（pp）：')
for k in ['ed2A', 'ed2A_E12', 'ed2B', 'ed2B_E1', 'base_rownorm18', 'base_magprune22', 'combo_2A_E12_plus_f26', 'ref_plus2000', 'ref_lr1e-4_plus2000']:
    ds = [(C[k][s_].sum() - C['original'][s_].sum()) / lens[s_].sum() * 100 for s_ in S]
    print(f'  {k:24s} {(C[k].sum()-C["original"].sum())/lens.sum()*100:+.2f}  [{np.percentile(ds,2.5):+.2f}, {np.percentile(ds,97.5):+.2f}]')
