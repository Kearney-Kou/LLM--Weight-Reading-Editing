"""Supplementary experiment (post hoc; settings, predictions and decision rules in 结果/预测_第三版补充实验.md): test of the noise hypothesis.
Apply the frozen H1 (heur.py, with / without compensation) to networks that were trained further, evaluate on B and C; differences are relative to "the network being edited itself".
Scoring and the storage of correct frames per utterance are the same as confirm_eval.py; the bootstrap is the same as summarize_confirm.py (rng seed 0, 2000 times, by utterance / by speaker).
Run from the 3_评估_勿给编辑者 root:
    python3 代码/noise_test.py eval        # compute correct frames per utterance → 结果/noise_counts.npz (can resume after interruption)
    python3 代码/noise_test.py summary     # bootstrap, row-norm profiles → 结果/noise_test_out.txt, 结果/noise_summary.json"""
import numpy as np, torch, sys, os, json
torch.set_num_threads(2)
E = '网络/'
L = lambda p: {k: v.astype(np.float32) for k, v in np.load(p).items()}
NETS = {'n1': 'subject_net.npz',                                   # consistency check: should reproduce the reported numbers
        'ss_lr1e-4': 'reference_samestate_lr1e-4_plus2000steps.npz',
        'fresh_lr1e-4': 'reference_lr1e-4_plus2000steps.npz',
        'cont2000': 'reference_plus2000steps.npz',
        'cont4000': 'reference_plus4000steps.npz'}
OUT = '结果/noise_counts.npz'

def profile(w):
    rn = np.linalg.norm(w['W1'].reshape(81, 28, 1024), axis=2); return rn.mean(1), rn

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
    for tag, f in NETS.items():
        w0 = L(E + f); m, offs = rownorm_mask(w0)
        print(tag, 'H1 选中偏移', offs, flush=True)
        variants = {'orig': w0, 'H1': H1(w0), 'H1nocomp': zero_comp(w0, m, comp=False)}
        for v, w in variants.items():
            for nm, D in [('B', B), ('C', C)]: run(f'{tag}_{v}@{nm}', w, D)

if sys.argv[1] == 'summary':
    sys.path.insert(0, '代码'); from heur import rownorm_mask
    r = np.load(OUT); ref = np.load('结果/confirm_counts.npz')
    Bd = np.load('数据/dev_B.npz'); Cd = np.load('数据/test_C.npz')
    info = {'B': (Bd['lens'].astype(float), np.array([n.split('-')[0] for n in Bd['names']])),
            'C': (Cd['lens'].astype(float), np.array([n.split('-')[0] for n in Cd['names']]))}
    rng = np.random.default_rng(0); boots = {}
    for k, (Ln, spk) in info.items():          # same sampling order as summarize_confirm.py
        U = len(Ln); us = np.unique(spk); ix = {s: np.where(spk == s)[0] for s in us}
        boots[k] = ([rng.integers(0, U, U) for _ in range(2000)], [np.concatenate([ix[s] for s in rng.choice(us, len(us))]) for _ in range(2000)])
    J = {}; lines = []
    P = lambda *a: (lines.append(' '.join(str(x) for x in a)), print(*a))
    P('一致性检查（被试网络，应与 confirm_counts.npz 逐句完全一致）：')
    for nm in ['B', 'C']:
        P(f'  n1 原网络 @{nm} 与 confirm_counts 中 n1_original@{nm} 逐句一致：', bool(np.array_equal(r[f'n1_orig@{nm}'], ref[f'n1_original@{nm}'])))
        for v, rk in [('H1nocomp', 'n1_H1_nocomp')]:
            if f'{rk}@{nm}' in ref.files: P(f'  n1 {v} @{nm} 与 {rk}@{nm} 逐句一致：', bool(np.array_equal(r[f'n1_{v}@{nm}'], ref[f'{rk}@{nm}'])))
    if 'n1_H1@C' in ref.files: P('  n1 H1 @C 与 n1_H1@C 逐句一致：', bool(np.array_equal(r['n1_H1@C'], ref['n1_H1@C'])))
    old = np.load('结果/per_utt_correct.npz')       # earlier results on B
    for mine, theirs, src in [('n1_H1@B', 'base_rownorm18', old), ('cont2000_orig@B', 'ref_plus2000', old), ('cont4000_orig@B', 'ref_plus4000', old),
                              ('fresh_lr1e-4_orig@B', 'ref_lr1e-4_plus2000', old), ('cont2000_orig@C', 'n1_ref_plus2000@C', ref),
                              ('fresh_lr1e-4_orig@C', 'n1_ref_lr1e-4@C', ref), ('ss_lr1e-4_orig@B', 'n1_ref_samestate_lr1e-4@B', ref),
                              ('ss_lr1e-4_orig@C', 'n1_ref_samestate_lr1e-4@C', ref)]:
        P(f'  {mine} 与已存的 {theirs} 逐句一致：', bool(np.array_equal(r[mine], src[theirs])))
    P('')
    for tag, f in NETS.items():
        w = L(E + f); prof, rn = profile(w); dt = np.arange(-40, 41)
        floor = float(prof[(np.abs(dt) >= 30) & (np.abs(dt) <= 38)].mean()); med26 = float(np.median(prof[np.abs(dt) >= 26]))
        J[tag] = dict(profile=prof.tolist(), floor_30_38=floor, median_ge26=med26, peak=float(prof.max()), offsets=rownorm_mask(w)[1])
        P(f'{tag:13s} 远端底 |Δt|30..38 均值 {floor:.4f}  |Δt|≥26 中位数 {med26:.4f}  峰值 {prof.max():.3f}  H1 选中 {J[tag]["offsets"]}')
        for nm in ['B', 'C']:
            Ln = info[nm][0]; b = r[f'{tag}_orig@{nm}']; J[tag][f'acc@{nm}'] = float(b.sum() / Ln.sum() * 100)
            for v in ['H1', 'H1nocomp']:
                a = r[f'{tag}_{v}@{nm}']; d = (a.sum() - b.sum()) / Ln.sum() * 100
                u = [(a[s].sum() - b[s].sum()) / Ln[s].sum() * 100 for s in boots[nm][0]]
                p = [(a[s].sum() - b[s].sum()) / Ln[s].sum() * 100 for s in boots[nm][1]]
                J[tag][f'{v}@{nm}'] = dict(d=float(d), utt=[float(np.percentile(u, 2.5)), float(np.percentile(u, 97.5))], spk=[float(np.percentile(p, 2.5)), float(np.percentile(p, 97.5))])
                P(f'    {nm}: 原网络 {b.sum() / Ln.sum() * 100:6.2f}  {v:9s} Δ {d:+.2f}  utt-CI [{np.percentile(u, 2.5):+.2f}, {np.percentile(u, 97.5):+.2f}]  spk-CI [{np.percentile(p, 2.5):+.2f}, {np.percentile(p, 97.5):+.2f}]')
    P('\n按预先写定的判断规则（g ≤ 0.5·g0 视为"明显变小"）：')
    f0 = J['n1']['floor_30_38']
    for tag in ['ss_lr1e-4', 'fresh_lr1e-4', 'cont2000', 'cont4000']:
        rat = {f'{v}@{nm}': J[tag][f'{v}@{nm}']['d'] / J['n1'][f'{v}@{nm}']['d'] for v in ['H1', 'H1nocomp'] for nm in ['B', 'C']}
        J[tag]['ratio_to_n1'] = rat; J[tag]['floor_change'] = J[tag]['floor_30_38'] / f0 - 1
        P(f'  {tag:13s} g/g0：' + '，'.join(f'{k} {x:.2f}' for k, x in rat.items()) + f'；远端底相对被试 {J[tag]["floor_change"]:+.1%}')
    json.dump(J, open('结果/noise_summary.json', 'w'), indent=1)
    open('结果/noise_test_out.txt', 'w').write('\n'.join(lines) + '\n')
