"""第五版补充分析 2（post hoc，设置、预测和判断规则见 结果/预测_第五版补充分析.md）：H1 在训练数据上的效果。
在 3_评估_勿给编辑者 根目录下运行：
    python3 代码/train_set_h1.py check <chunk0> <chunk1> <chunk2>   # 用重建的顺序重跑被试训练前 200 步，核对 train_log.txt（loss 1.518，monA 0.5399）
    python3 代码/train_set_h1.py eval  <chunk0> <chunk1> <chunk2>   # → 结果/train_set_h1_counts.npz（可断点续算）
    python3 代码/train_set_h1.py summary <chunk0> <chunk1> <chunk2> # → 结果/train_set_h1_out.txt、train_set_h1.json
训练顺序：被试（train_subject.py）和 lr 1e-4 长训练（train_reference_lowlr_long.py）都是 torch.manual_seed(0) → 建网络 → randperm，所以顺序相同。
被试看过 perm[:9,625,600]（各一次）；第 10,000 步检查点在此之后又看了 perm[:10,240,000]。"""
import numpy as np, torch, sys, os, json
torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_subject_lib import TrainFrames
mode, paths = sys.argv[1], sys.argv[2:]
NTR = 15203231; SUBJ = 9400 * 1024; LONG = 10000 * 1024; NS = 1_000_000
def make_net():
    return torch.nn.Sequential(torch.nn.Linear(2268, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 42))
def order():
    torch.manual_seed(0); net = make_net(); return net, torch.randperm(NTR)
tr = TrainFrames(paths); assert len(tr) == NTR

if mode == 'check':
    from wlib import Frames
    np.random.seed(0)
    net, perm = order()
    dA = Frames('数据/dev_A.npz'); mon = torch.from_numpy(np.random.default_rng(1).choice(len(dA), 100000, replace=False))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01); lossf = torch.nn.CrossEntropyLoss()
    for s in range(200):
        idx = perm[s * 1024:(s + 1) * 1024]; loss = lossf(net(tr.windows(idx)), tr.y[idx]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): a = (net(dA.windows(mon)).argmax(1) == dA.y[mon]).float().mean().item()
    print(f'重跑第 200 步：loss {loss.item():.3f}  monA {a:.4f}（train_log.txt：loss 1.518  monA 0.5399）', flush=True)
    sys.exit(0)

_, perm = order(); perm = perm.numpy()
L = np.concatenate([np.load(p)['lens'] for p in paths]); utt_of = np.repeat(np.arange(len(L)), L)
rng = np.random.default_rng(0)
SETS = {'n1_seen': np.sort(rng.choice(perm[:SUBJ], NS, replace=False)), 'n1_unseen': np.sort(rng.choice(perm[SUBJ:], NS, replace=False)),
        'l10k_seen': np.sort(rng.choice(perm[:LONG], NS, replace=False)), 'l10k_unseen': np.sort(rng.choice(perm[LONG:], NS, replace=False))}
OUT = '结果/train_set_h1_counts.npz'
L32 = lambda p: {k: v.astype(np.float32) for k, v in np.load(p).items()}

@torch.no_grad()
def per_utt(w, X, y, utt, nU, bs=16384):
    t = {k: torch.as_tensor(v) for k, v in w.items()}; ok = np.zeros(nU); ce = np.zeros(nU)
    for i in range(0, len(y), bs):
        x = X(i, min(i + bs, len(y)))
        lg = torch.relu(torch.relu(x @ t['W1'] + t['b1']) @ t['W2'] + t['b2']) @ t['W3'] + t['b3']
        yy = torch.as_tensor(y[i:i + bs]); u = utt[i:i + bs]
        np.add.at(ok, u, (lg.argmax(1) == yy).double().numpy()); np.add.at(ce, u, torch.nn.functional.cross_entropy(lg, yy, reduction='none').double().numpy())
    return ok, ce

if mode == 'eval':
    from heur import H1, zero_comp, rownorm_mask
    from wlib import Frames
    res = dict(np.load(OUT)) if os.path.exists(OUT) else {}
    nets = {}
    for tag, p in [('n1', '网络/subject_net.npz'), ('l10k', '网络/reference_lr1e-4_long_step10000.npz')]:
        w0 = L32(p); m, _ = rownorm_mask(w0)
        nets[tag] = {'orig': w0, 'H1': H1(w0), 'H1nocomp': zero_comp(w0, m, comp=False)}
    def run(key, w, X, y, utt, nU):
        if key + '_ok' in res: return
        ok, ce = per_utt(w, X, y, utt, nU); res[key + '_ok'] = ok; res[key + '_ce'] = ce
        np.savez(OUT, **res); print('done', key, f'acc {ok.sum() / len(y):.4f}  ce {ce.sum() / len(y):.4f}', flush=True)
    yt = tr.y.numpy()
    for tag in ['n1', 'l10k']:
        for cat in ['seen', 'unseen']:
            idx = SETS[f'{tag}_{cat}']; y = yt[idx]; utt = utt_of[idx]
            X = lambda a, b, idx=idx: tr.windows(torch.from_numpy(idx[a:b]))
            for v, w in nets[tag].items(): run(f'{tag}_{v}@train_{cat}', w, X, y, utt, len(L))
    for nm, path in [('B', '数据/dev_B.npz'), ('C', '数据/test_C.npz')]:
        D = Frames(path); y = D.y.numpy(); utt = D.utt.numpy()
        X = lambda a, b, D=D: D.windows(torch.arange(a, b))
        for tag in ['n1', 'l10k']:
            for v, w in nets[tag].items(): run(f'{tag}_{v}@{nm}', w, X, y, utt, len(D.lens))

if mode == 'summary':
    r = np.load(OUT); lines = []; P = lambda *a: (lines.append(' '.join(str(x) for x in a)), print(*a)); J = {}
    Bd = np.load('数据/dev_B.npz'); Cd = np.load('数据/test_C.npz')
    nfr = {'B': Bd['lens'].astype(float), 'C': Cd['lens'].astype(float)}
    for tag in ['n1', 'l10k']:
        for cat in ['seen', 'unseen']:
            nfr[f'{tag}_train_{cat}'] = np.bincount(utt_of[SETS[f'{tag}_{cat}']], minlength=len(L)).astype(float)
    rb = np.random.default_rng(0)
    boot = {'train': [rb.integers(0, len(L), len(L)) for _ in range(2000)], 'B': [rb.integers(0, len(nfr['B']), len(nfr['B'])) for _ in range(2000)],
            'C': [rb.integers(0, len(nfr['C']), len(nfr['C'])) for _ in range(2000)]}
    def stat(tag, v, ds):
        base = f'{tag}_orig@{ds}'; key = f'{tag}_{v}@{ds}'; n = nfr[ds if ds in 'BC' else f'{tag}_{ds}']; bs = boot['train' if ds.startswith('train') else ds]
        da = (r[key + '_ok'] - r[base + '_ok']); dc = (r[key + '_ce'] - r[base + '_ce'])
        acc = lambda s: da[s].sum() / n[s].sum() * 100; ce = lambda s: dc[s].sum() / n[s].sum()
        A = [acc(s) for s in bs]; C = [ce(s) for s in bs]
        return dict(acc0=float(r[base + '_ok'].sum() / n.sum() * 100), ce0=float(r[base + '_ce'].sum() / n.sum()), d_acc=float(da.sum() / n.sum() * 100),
                    d_acc_ci=[float(np.percentile(A, 2.5)), float(np.percentile(A, 97.5))], d_ce=float(dc.sum() / n.sum()),
                    d_ce_ci=[float(np.percentile(C, 2.5)), float(np.percentile(C, 97.5))])
    for tag, name in [('n1', '被试'), ('l10k', 'lr1e-4 第 10000 步')]:
        P(f'\n{name}：')
        for ds in ['train_seen', 'train_unseen', 'B', 'C']:
            for v in ['H1', 'H1nocomp']:
                s = stat(tag, v, ds); J[f'{tag}_{v}@{ds}'] = s
                P(f'  {ds:13s} 原网络 acc {s["acc0"]:6.2f}  ce {s["ce0"]:.4f}   {v:9s} Δacc {s["d_acc"]:+.2f} [{s["d_acc_ci"][0]:+.2f}, {s["d_acc_ci"][1]:+.2f}]  '
                  f'Δce {s["d_ce"]:+.4f} [{s["d_ce_ci"][0]:+.4f}, {s["d_ce_ci"][1]:+.4f}]')
        for v in ['H1', 'H1nocomp']:   # seen − unseen 的差（同一次重抽样）
            ks, ku = f'{tag}_{v}@train_seen', f'{tag}_{v}@train_unseen'; bs = f'{tag}_orig@train_seen', f'{tag}_orig@train_unseen'
            ns, nu = nfr[f'{tag}_train_seen'], nfr[f'{tag}_train_unseen']
            ds_ = r[ks + '_ok'] - r[bs[0] + '_ok']; du_ = r[ku + '_ok'] - r[bs[1] + '_ok']
            D = [(ds_[s].sum() / ns[s].sum() - du_[s].sum() / nu[s].sum()) * 100 for s in boot['train']]
            d = (ds_.sum() / ns.sum() - du_.sum() / nu.sum()) * 100; J[f'{tag}_{v}_seen_minus_unseen'] = dict(d=float(d), ci=[float(np.percentile(D, 2.5)), float(np.percentile(D, 97.5))])
            P(f'  {v:9s} g_seen − g_unseen = {d:+.2f} [{np.percentile(D, 2.5):+.2f}, {np.percentile(D, 97.5):+.2f}] pp')
    gs, gu = J['n1_H1@train_seen'], J['n1_H1@train_unseen']; dd = J['n1_H1_seen_minus_unseen']
    if gs['d_acc'] > 0 and gs['d_acc_ci'][0] > 0 and gs['d_acc'] >= 0.5 * gu['d_acc']: v = '支持噪声解释'
    elif (gs['d_acc'] <= 0 and gu['d_acc_ci'][0] > 0) or (gs['d_acc'] < 0.5 * gu['d_acc'] and (dd['ci'][1] < 0 or dd['ci'][0] > 0)): v = '支持过拟合解释'
    else: v = '结果不一致'
    P('\n按预先写定的判断规则（被试、带补偿的 H1、准确率）：', v); J['verdict'] = v
    json.dump(J, open('结果/train_set_h1.json', 'w'), indent=1, ensure_ascii=False)
    open('结果/train_set_h1_out.txt', 'w').write('\n'.join(lines) + '\n')
