"""第五版补充分析 1（post hoc，设置、预测和判断规则见 结果/预测_第五版补充分析.md）：远端权重位移的径向/切向分解。
在 3_评估_勿给编辑者 根目录下运行：python3 代码/displacement.py  → 结果/displacement_out.txt、结果/displacement.json
只读权重，不需要数据。"""
import numpy as np, json
dt = np.arange(-40, 41)
FAR = (np.abs(dt) >= 30) & (np.abs(dt) <= 38); NEAR = np.abs(dt) <= 10
W = lambda p: np.load(p)['W1'].astype(np.float64).reshape(81, 28, 1024)
w0 = W('网络/subject_net.npz')
NETS = [(f'lr1e-4 fresh, +{s}', f'网络/reference_lr1e-4_long_step{s}.npz', 1e-4, s) for s in [2000, 4000, 6000, 8000, 10000]] + [
        ('lr1e-4 same state, +2000', '网络/reference_samestate_lr1e-4_plus2000steps.npz', 1e-4, 2000),
        ('lr1e-3 same state, +2000', '网络/reference_plus2000steps.npz', 1e-3, 2000),
        ('lr1e-3 same state, +4000', '网络/reference_plus4000steps.npz', 1e-3, 4000)]
lines = []; P = lambda *a: (lines.append(' '.join(str(x) for x in a)), print(*a))
def decomp(a, b):
    a = a.ravel(); d = b.ravel() - a; n = np.sqrt(a.size)
    r = a @ d / np.linalg.norm(a); t = np.sqrt(max(d @ d - r * r, 0.0))
    return dict(w_rms=float(np.linalg.norm(a) / n), d_rms=float(np.linalg.norm(d) / n), radial=float(r / n), tangential=float(t / n),
                norm_change=float(np.linalg.norm(b.ravel()) / np.linalg.norm(a) - 1))
J = {}
P('每个权重的量：w_rms = 被试权重的 RMS；d_rms = 位移 RMS；radial = 径向分量（负 = 朝 0）；tangential = 切向分量；wd = weight decay 能解释的径向位移；η√T = 纯随机游走的预期位移')
for name, path, lr, T in NETS:
    w = W(path); J[name] = {}
    for reg, m in [('far', FAR), ('near', NEAR)]:
        D = decomp(w0[m], w[m]); D['wd'] = float(-(1 - (1 - 0.01 * lr) ** T) * D['w_rms']); D['eta_sqrtT'] = lr * np.sqrt(T)
        rows0 = w0[m].reshape(-1, 1024); rows = w[m].reshape(-1, 1024)
        rr = np.einsum('ij,ij->i', rows0, rows - rows0) / np.linalg.norm(rows0, axis=1)
        D['frac_rows_inward'] = float((rr < 0).mean()); J[name][reg] = D
        P(f'{name:26s} {reg:4s}  w_rms {D["w_rms"]:.4f}  d_rms {D["d_rms"]:.4f}  radial {D["radial"]:+.4f}  tangential {D["tangential"]:.4f}  '
          f'wd {D["wd"]:+.5f}  radial/wd {D["radial"] / D["wd"]:5.1f}  |radial|/tangential {abs(D["radial"]) / D["tangential"]:.2f}  '
          f'η√T {D["eta_sqrtT"]:.4f}  范数变化 {D["norm_change"]:+.1%}  朝 0 的行 {D["frac_rows_inward"]:.0%}')
f10, f2 = J['lr1e-4 fresh, +10000']['far'], J['lr1e-4 fresh, +2000']['far']
ratio = f10['d_rms'] / f2['d_rms']; J['ratio_10k_2k_far'] = ratio
J['ratio_10k_2k_near'] = J['lr1e-4 fresh, +10000']['near']['d_rms'] / J['lr1e-4 fresh, +2000']['near']['d_rms']
P(f'\n远端位移 RMS 之比（第 10000 步 / 第 2000 步）：{ratio:.2f}（随机游走 √5 = 2.24，匀速漂移 5）；中心附近：{J["ratio_10k_2k_near"]:.2f}')
r, t, wd = f10['radial'], f10['tangential'], f10['wd']
if r < 0 and abs(r) >= 3 * abs(wd) and abs(r) < 0.5 * t: v1 = '有一个弱的、指向 0 的拉力，但被随机运动主导'
elif r < 0 and abs(r) >= 0.5 * t: v1 = '有明显的拉力'
elif abs(r) < 3 * abs(wd): v1 = '除了 weight decay 之外检测不到拉力'
else: v1 = '其他（径向分量不朝 0）'
v2 = '位移按 √T 增长，和随机游走一致' if 1.8 <= ratio <= 2.7 else ('更像漂移' if ratio >= 3.5 else '介于两者之间')
P('按预先写定的判断规则：', v1, '；', v2); J['verdict'] = [v1, v2]
json.dump(J, open('结果/displacement.json', 'w'), indent=1, ensure_ascii=False)
open('结果/displacement_out.txt', 'w').write('\n'.join(lines) + '\n')
