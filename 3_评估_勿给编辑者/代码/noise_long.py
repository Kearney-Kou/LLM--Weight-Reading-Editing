"""第四版补充实验（post hoc，设置和预测见 结果/预测_第四版补充实验.md）：H1 在 lr 1e-4 训练更久的各检查点上的收益。
评估方法与 noise_test.py 完全相同（逐句正确帧数；按句子 / 按说话人 bootstrap 2000 次，rng 种子 0）。
在 3_评估_勿给编辑者 根目录下运行：
    python3 代码/noise_long.py eval      # → 结果/noise_long_counts.npz（可断点续算）
    python3 代码/noise_long.py summary   # → 结果/noise_long_out.txt、结果/noise_long_summary.json"""
import numpy as np, torch, sys, os, json
torch.set_num_threads(2)
L = lambda p: {k: v.astype(np.float32) for k, v in np.load(p).items()}
STEPS = [2000, 4000, 6000, 8000, 10000]
OUT = '结果/noise_long_counts.npz'
path = lambda s: f'网络/reference_lr1e-4_long_step{s}.npz'

if sys.argv[1] == 'eval':
    from wlib import Frames, predict
    from heur import H1, zero_comp, rownorm_mask
    B = Frames('数据/dev_B.npz'); C = Frames('数据/test_C.npz')
    res = dict(np.load(OUT)) if os.path.exists(OUT) else {}
    def run(key, w, D):
        if key in res: return
        ok = (predict(w, D) == D.y).double()
        res[key] = torch.zeros(len(D.lens), dtype=torch.float64).index_add_(0, D.utt, ok).numpy()
        np.savez(OUT, **res); print('done', key, round(res[key].sum() / D.lens.sum(), 4), flush=True)
    for s in STEPS:
        w0 = L(path(s)); m, offs = rownorm_mask(w0); print(s, 'H1 选中偏移', offs, flush=True)
        for v, w in {'orig': w0, 'H1': H1(w0), 'H1nocomp': zero_comp(w0, m, comp=False)}.items():
            for nm, D in [('B', B), ('C', C)]: run(f's{s}_{v}@{nm}', w, D)

if sys.argv[1] == 'summary':
    sys.path.insert(0, '代码'); from heur import rownorm_mask
    r = np.load(OUT); prev = np.load('结果/noise_counts.npz'); ref2k = np.load('网络/reference_lr1e-4_plus2000steps.npz')
    Bd = np.load('数据/dev_B.npz'); Cd = np.load('数据/test_C.npz')
    info = {'B': (Bd['lens'].astype(float), np.array([n.split('-')[0] for n in Bd['names']])),
            'C': (Cd['lens'].astype(float), np.array([n.split('-')[0] for n in Cd['names']]))}
    rng = np.random.default_rng(0); boots = {}
    for k, (Ln, spk) in info.items():          # 与 summarize_confirm.py / noise_test.py 相同的抽样顺序
        U = len(Ln); us = np.unique(spk); ix = {s: np.where(spk == s)[0] for s in us}
        boots[k] = ([rng.integers(0, U, U) for _ in range(2000)], [np.concatenate([ix[s] for s in rng.choice(us, len(us))]) for _ in range(2000)])
    lines = []; P = lambda *a: (lines.append(' '.join(str(x) for x in a)), print(*a))
    w2 = L(path(2000))
    P('一致性检查：新跑的第 2000 步 与 reference_lr1e-4_plus2000steps.npz 的最大绝对差',
      {k: float(np.abs(w2[k] - ref2k[k]).max()) for k in w2})
    for nm in 'BC':
        for v in ['orig', 'H1', 'H1nocomp']:
            P(f'  第 2000 步 {v}@{nm} 与第三版 fresh_lr1e-4_{v}@{nm} 逐句一致：', bool(np.array_equal(r[f's2000_{v}@{nm}'], prev[f'fresh_lr1e-4_{v}@{nm}'])))
    sub = np.linalg.norm(L('网络/subject_net.npz')['W1'].reshape(81, 28, 1024), axis=2).mean(1); dt = np.arange(-40, 41)
    far = (np.abs(dt) >= 30) & (np.abs(dt) <= 38); near = np.abs(dt) <= 10
    J = {'subject_profile': sub.tolist()}
    for s in STEPS:
        w = L(path(s)); prof = np.linalg.norm(w['W1'].reshape(81, 28, 1024), axis=2).mean(1)
        J[s] = dict(profile=prof.tolist(), floor_change=float(prof[far].mean() / sub[far].mean() - 1),
                    near_change=float(prof[near].mean() / sub[near].mean() - 1), offsets=rownorm_mask(w)[1])
        P(f'\n第 {s} 步：远端底相对被试 {J[s]["floor_change"]:+.1%}，|Δt|≤10 相对被试 {J[s]["near_change"]:+.1%}，H1 选中 {J[s]["offsets"]}')
        for nm in 'BC':
            Ln = info[nm][0]; b = r[f's{s}_orig@{nm}']; J[s][f'acc@{nm}'] = float(b.sum() / Ln.sum() * 100)
            for v in ['H1', 'H1nocomp']:
                a = r[f's{s}_{v}@{nm}']; d = (a.sum() - b.sum()) / Ln.sum() * 100
                u = [(a[x].sum() - b[x].sum()) / Ln[x].sum() * 100 for x in boots[nm][0]]
                p = [(a[x].sum() - b[x].sum()) / Ln[x].sum() * 100 for x in boots[nm][1]]
                J[s][f'{v}@{nm}'] = dict(d=float(d), utt=[float(np.percentile(u, 2.5)), float(np.percentile(u, 97.5))], spk=[float(np.percentile(p, 2.5)), float(np.percentile(p, 97.5))])
                P(f'    {nm}: 原网络 {b.sum() / Ln.sum() * 100:6.2f}  {v:9s} Δ {d:+.2f}  utt-CI [{np.percentile(u, 2.5):+.2f}, {np.percentile(u, 97.5):+.2f}]  spk-CI [{np.percentile(p, 2.5):+.2f}, {np.percentile(p, 97.5):+.2f}]')
    w0 = L('网络/subject_net.npz')['W1'].reshape(81, 28, 1024); w10 = L(path(10000))['W1'].reshape(81, 28, 1024)
    rms = lambda x: float(np.sqrt((x ** 2).mean()))
    J['displacement'] = dict(far_rms_weight=rms(w0[far]), far_rms_change=rms(w10[far] - w0[far]), near_rms_weight=rms(w0[near]), near_rms_change=rms(w10[near] - w0[near]))
    P('\n被试到第 10000 步，单个权重的 RMS 大小和 RMS 位移：', {k: round(v, 4) for k, v in J['displacement'].items()})
    last = [J[10000][f'{v}@{nm}'] for v in ['H1', 'H1nocomp'] for nm in 'BC']
    if all(x['utt'][0] > 0 for x in last): verdict = '优化在这段训练里没有去掉障碍（4 格区间全部 > 0）'
    elif all(x['d'] <= 0.02 for x in last) or all(x['utt'][0] <= 0 <= x['utt'][1] for x in last): verdict = '优化最终也消掉了这份损害'
    else: verdict = '结果不一致'
    P('\n按预先写定的判断规则（第 10000 步，4 格）：', verdict); J['verdict'] = verdict
    json.dump(J, open('结果/noise_long_summary.json', 'w'), indent=1, ensure_ascii=False)
    open('结果/noise_long_out.txt', 'w').write('\n'.join(lines) + '\n')
