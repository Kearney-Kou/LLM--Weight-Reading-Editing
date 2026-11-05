"""评估（只在编辑者交卷后运行一次）。用法：
    python3 代码/evaluate.py <修改后的网络.npz> [更多 .npz ...]   （在 3_评估_勿给编辑者 根目录下运行）
在考试集 B 上比较：原网络、修改后的网络、继续训练 500/1000/2000/4000 步的参照、降学习率训练 2000 步的参照、随机修改对照（每个提交 20 份）。
差值按"句子"做 bootstrap（2000 次）给 95% 置信区间（同一句话里的帧高度相关，不能当独立样本）。
报告写到 评估报告_<文件名>.md。"""
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 3_评估_勿给编辑者 根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wlib import Frames, predict, NAMES

def load(p):
    d = np.load(p); return {k: d[k].astype(np.float32) for k in NAMES}

def per_utt(w, data):
    ok = (predict(w, data) == data.y).double()
    return torch.zeros(len(data.lens), dtype=torch.float64).index_add_(0, data.utt, ok).numpy()

def boot_ci(c_new, c_old, lens, n=2000, seed=0):
    rng = np.random.default_rng(seed); U = len(lens); ds = []
    for _ in range(n):
        s = rng.integers(0, U, U); ds.append((c_new[s].sum() - c_old[s].sum()) / lens[s].sum())
    return np.percentile(ds, [2.5, 97.5])

def random_like(w0, w1, rng):
    """同样的每个张量改动个数、同样的改动范数，位置和方向随机。"""
    out = {}
    for k in NAMES:
        d = w1[k] - w0[k]; nz = np.flatnonzero(d); r = w0[k].copy()
        if len(nz):
            pos = rng.choice(w0[k].size, len(nz), replace=False)
            v = rng.standard_normal(len(nz)).astype(np.float32); v *= np.linalg.norm(d) / np.linalg.norm(v)
            r.reshape(-1)[pos] += v
        out[k] = r
    return out

def main(paths):
    w0 = load(os.path.join(HERE, '网络/subject_net.npz'))
    refs = [(f'继续训练 {n} 步（lr 1e-3，{n*1024:,} 帧）', load(os.path.join(HERE, f'网络/reference_plus{n}steps.npz'))) for n in [500, 1000, 2000, 4000]]
    refs.append(('lr 降到 1e-4 再训练 2000 步', load(os.path.join(HERE, '网络/reference_lr1e-4_plus2000steps.npz'))))
    B = Frames(os.path.join(HERE, '数据/dev_B.npz')); lens = B.lens.astype(np.float64); N = lens.sum()
    c0 = per_utt(w0, B); crs = [(n, per_utt(wr, B)) for n, wr in refs]
    for p in paths:
        w1 = load(p); c1 = per_utt(w1, B)
        lines = [f'# 评估报告：{os.path.basename(p)}', '', f'考试集 B：{len(lens)} 句，{int(N)} 帧', '',
                 '| 网络 | B 上准确率 | 相对原网络 | 95% 置信区间（按句子 bootstrap） |', '|---|---|---|---|',
                 f'| 原网络（被试） | {c0.sum()/N:.4f} | — | — |']
        lo, hi = boot_ci(c1, c0, lens)
        lines.append(f'| **修改后** | **{c1.sum()/N:.4f}** | **{(c1.sum()-c0.sum())/N:+.4f}** | [{lo:+.4f}, {hi:+.4f}] |')
        for n, cr in crs:
            lo, hi = boot_ci(cr, c0, lens)
            lines.append(f'| 参照：{n} | {cr.sum()/N:.4f} | {(cr.sum()-c0.sum())/N:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
        rng = np.random.default_rng(1); rs = [per_utt(random_like(w0, w1, rng), B).sum() / N - c0.sum() / N for _ in range(int(os.environ.get('N_RANDOM', '20')))]
        lines.append(f'| 对照：随机修改 ×{len(rs)}（同个数、同范数） | 均值 {c0.sum()/N+np.mean(rs):.4f} | 均值 {np.mean(rs):+.4f}，范围 [{min(rs):+.4f}, {max(rs):+.4f}] | — |')
        lines += ['', '## 修改的大小', '', '| 张量 | 改动个数 | 占比 | 改动范数 | 原张量范数 |', '|---|---|---|---|---|']
        tot = 0
        for k in NAMES:
            d = w1[k] - w0[k]; n = int((d != 0).sum()); tot += n
            lines.append(f'| {k} | {n} | {n/d.size:.2%} | {np.linalg.norm(d):.4f} | {np.linalg.norm(w0[k]):.4f} |')
        lines.append(f'| 合计 | {tot} | {tot/sum(w0[k].size for k in NAMES):.2%} | | |')
        out = os.path.join(HERE, '结果', f'评估报告_{os.path.splitext(os.path.basename(p))[0]}.md')
        open(out, 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines)); print('写入', out)

if __name__ == '__main__':
    main(sys.argv[1:])
