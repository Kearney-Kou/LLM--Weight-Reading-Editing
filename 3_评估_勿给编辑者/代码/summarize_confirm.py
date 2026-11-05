import numpy as np
r = np.load('结果/confirm_counts.npz')
B = np.load('数据/dev_B.npz'); C = np.load('数据/test_C.npz')
info = {'B': (B['lens'].astype(float), np.array([n.split('-')[0] for n in B['names']])), 'C': (C['lens'].astype(float), np.array([n.split('-')[0] for n in C['names']]))}
import json
rng = np.random.default_rng(0); J = {}
boots = {}
for k, (L, spk) in info.items():
    U = len(L); us = np.unique(spk); ix = {s: np.where(spk == s)[0] for s in us}
    boots[k] = ([rng.integers(0, U, U) for _ in range(2000)], [np.concatenate([ix[s] for s in rng.choice(us, len(us))]) for _ in range(2000)], len(us))
def row(key, base, store=None):
    nm = key.split('@')[1]; L = info[nm][0]; a = r[key]; b = r[base]
    d = (a.sum() - b.sum()) / L.sum() * 100
    u = [(a[s].sum() - b[s].sum()) / L[s].sum() * 100 for s in boots[nm][0]]
    p = [(a[s].sum() - b[s].sum()) / L[s].sum() * 100 for s in boots[nm][1]]
    J[store or key] = dict(acc=a.sum()/L.sum()*100, d=d, utt=[np.percentile(u,2.5), np.percentile(u,97.5)], spk=[np.percentile(p,2.5), np.percentile(p,97.5)])
    return f'{key:28s} acc {a.sum()/L.sum()*100:6.2f}  Δ {d:+.2f}  utt-CI [{np.percentile(u,2.5):+.2f}, {np.percentile(u,97.5):+.2f}]  spk-CI [{np.percentile(p,2.5):+.2f}, {np.percentile(p,97.5):+.2f}]'
for nm in ['B', 'C']: print(f'{nm}: {len(info[nm][0])} 句, {int(info[nm][0].sum())} 帧, {boots[nm][2]} 位说话人')
for key in sorted(r.files):
    pre, nm = key.split('@'); net = pre.split('_')[0]
    if pre.endswith('original'): print(f'{key:28s} acc {r[key].sum()/info[nm][0].sum()*100:6.2f}'); continue
    print(row(key, f'{net}_original@{nm}'))
for nm in ['B','C']: J[f'n1_original@{nm}'] = dict(acc=r[f'n1_original@{nm}'].sum()/info[nm][0].sum()*100)
for s_ in ['s1','s2']:
    for nm in ['B','C']: J[f'{s_}_original@{nm}'] = dict(acc=r[f'{s_}_original@{nm}'].sum()/info[nm][0].sum()*100)
if 'n1_H1@C' in r.files and 'n1_ed2A_E12@C' in r.files: print('C: H1 − ed2A_E12 →', row('n1_H1@C', 'n1_ed2A_E12@C', store='H1_minus_E12@C').split('Δ')[1])
# JSON 键 H1_minus_E12@C：C 上 H1 − ed2A_E12 的差
json.dump(J, open('结果/confirm_summary.json', 'w'), indent=1)
