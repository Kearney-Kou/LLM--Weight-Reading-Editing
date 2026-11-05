"""How to run: from the 3_评估_勿给编辑者 root, run python3 代码/final_table.py
Main results table: accuracy of all networks on B, change relative to the original network, utterance-paired bootstrap 95% intervals (2000 times, seed 0).
Also saves the number of correct frames per utterance (per_utt_correct.npz) for plotting, and saves the weights of the baseline networks (baselines/)."""
import numpy as np, torch, os
from wlib import Frames, predict, NAMES
torch.set_num_threads(2)
w0 = {k: v.astype(np.float32) for k, v in np.load('网络/subject_net.npz').items()}
A = Frames('数据/dev_A.npz'); B = Frames('数据/dev_B.npz')
idx = torch.from_numpy(np.sort(np.random.default_rng(0).choice(len(A), 40000, replace=False)))
XA = A.windows(idx).numpy()
def h1mean(w): return np.maximum(XA @ w['W1'] + w['b1'], 0).mean(0)
m0 = h1mean(w0)
def zc(mask):
    w = {k: v.copy() for k, v in w0.items()}; w['W1'][mask] = 0
    w['b2'] = w0['b2'] + (m0 - h1mean(w)) @ w0['W2']; return w
L = lambda p: {k: v.astype(np.float32) for k, v in np.load(p).items()}
nets = {'original': w0, 'ed2A': L('网络/edited_2A.npz'), 'ed2B': L('网络/edited_2B.npz'), 'ed2B_E1': L('网络/edited_2B_E1only.npz')}
e = nets['ed2A']; e12 = {k: v.copy() for k, v in e.items()}; e12['W3'] = w0['W3'].copy(); e12['b3'] = w0['b3'].copy(); nets['ed2A_E12'] = e12
rn = np.linalg.norm(w0['W1'].reshape(81, 28, 1024), axis=2).mean(1); ts = np.argsort(rn)[:18]
m = np.zeros((81, 28, 1024), bool); m[ts] = True; nets['base_rownorm18'] = zc(m.reshape(2268, 1024))
thr = np.sort(np.abs(w0['W1']).ravel())[516096 - 1]; nets['base_magprune22'] = zc(np.abs(w0['W1']) <= thr)
m = np.zeros((81, 28, 1024), bool); m[:, 26] = True
w5 = {k: v.copy() for k, v in e12.items()}; w5['W1'][m.reshape(2268, 1024)] = 0
w5['b2'] = e12['b2'] + (h1mean(e12) - h1mean(w5)) @ w0['W2']; nets['combo_2A_E12_plus_f26'] = w5
for n in [500, 1000, 2000, 4000]: nets[f'ref_plus{n}'] = L(f'网络/reference_plus{n}steps.npz')
nets['ref_lr1e-4_plus2000'] = L('网络/reference_lr1e-4_plus2000steps.npz')
os.makedirs('网络/baselines', exist_ok=True)
for k in ['base_rownorm18', 'base_magprune22', 'combo_2A_E12_plus_f26', 'ed2A_E12']: np.savez(f'网络/baselines/{k}.npz', **nets[k])
lens = B.lens.astype(np.float64); N = lens.sum(); U = len(lens)
C = {}
for k, w in nets.items():
    ok = (predict(w, B) == B.y).double()
    C[k] = torch.zeros(U, dtype=torch.float64).index_add_(0, B.utt, ok).numpy()
np.savez('结果/per_utt_correct.npz', lens=lens, **C)
rng = np.random.default_rng(0); S = [rng.integers(0, U, U) for _ in range(2000)]
print('| network | B acc | delta (pp) | 95% CI (pp) |')
for k in nets:
    d = (C[k].sum() - C['original'].sum()) / N
    if k == 'original': print(f'| {k} | {C[k].sum()/N:.4f} | — | — |'); continue
    ds = [(C[k][s].sum() - C['original'][s].sum()) / lens[s].sum() for s in S]
    lo, hi = np.percentile(ds, [2.5, 97.5])
    print(f'| {k} | {C[k].sum()/N:.4f} | {100*d:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |', flush=True)
print('rownorm18 offsets', sorted((ts - 40).tolist()))
