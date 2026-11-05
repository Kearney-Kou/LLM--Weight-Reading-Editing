"""Confirmatory test (as pre-registered). For each (network, test set), compute the number of correct frames per utterance and save it to confirm_counts.npz."""
import numpy as np, torch, sys, os
from wlib import Frames, predict
from heur import H1, H2, H3, H2cal, zero_comp, rownorm_mask, feat_mask, min_eta_feature
torch.set_num_threads(2)
E = '网络/'
L = lambda p: {k: v.astype(np.float32) for k, v in np.load(p).items()}
B = Frames('数据/dev_B.npz'); C = Frames('数据/test_C.npz')
out = '结果/confirm_counts.npz'; res = dict(np.load(out)) if os.path.exists(out) else {}
def run(tag, w, sets):
    for nm, D in sets:
        key = f'{tag}@{nm}'
        if key in res: continue
        ok = (predict(w, D) == D.y).double()
        res[key] = torch.zeros(len(D.lens), dtype=torch.float64).index_add_(0, D.utt, ok).numpy()
        np.savez(out, **res); print('done', key, round(res[key].sum() / D.lens.sum(), 4), flush=True)
part = sys.argv[1]
if part == 'net1':
    w0 = L(E + 'subject_net.npz')
    print('H2 选中特征', min_eta_feature(), '; H1 选中偏移', rownorm_mask(w0)[1], flush=True)
    nets = {'original': w0, 'ed2A': L(E + 'edited_2A.npz'), 'ed2A_E12': L(E + 'baselines/ed2A_E12.npz'), 'ed2B': L(E + 'edited_2B.npz'),
            'ed2B_E1': L(E + 'edited_2B_E1only.npz'), 'magprune22': L(E + 'baselines/base_magprune22.npz'), 'combo_2A_f26': L(E + 'baselines/combo_2A_E12_plus_f26.npz'),
            'H1': H1(w0), 'H3': H3(w0), 'H2cal': H2cal(w0),
            'ref_plus2000': L(E + 'reference_plus2000steps.npz'), 'ref_lr1e-4': L(E + 'reference_lr1e-4_plus2000steps.npz')}
    for k, w in nets.items(): run('n1_' + k, w, [('C', C)])
    # Added variants (evaluated on both B and C): 2B's full non-LLM baseline; mechanism separation (delete only, no compensation)
    extra = {'H2cal': nets['H2cal'], 'H1_nocomp': zero_comp(w0, rownorm_mask(w0)[0], comp=False), 'H2_nocomp': zero_comp(w0, feat_mask(26), comp=False)}
    run('n1_original', w0, [('B', B)])
    for k, w in extra.items(): run('n1_' + k, w, [('B', B), ('C', C)])
if part.startswith('seed'):
    s = part[4:]; w0 = L(f'网络/subject_seed{s}.npz')
    print(f'seed {s}: H2 选中特征', min_eta_feature(), '; H1 选中偏移', rownorm_mask(w0)[1], flush=True)
    for k, w in {'original': w0, 'H1': H1(w0), 'H2': H2(w0), 'H3': H3(w0), 'H2cal': H2cal(w0)}.items():
        run(f's{s}_' + k, w, [('B', B), ('C', C)])
if part == 'freshref':
    run('n1_ref_fresh_lr1e-3', L('网络/reference_fresh_lr1e-3_plus2000steps.npz'), [('B', B), ('C', C)])
if part == 'samestate':   # Done afterwards (not pre-registered): keep the original AdamW state, lr 1e-4, +2000 steps
    run('n1_ref_samestate_lr1e-4', L('网络/reference_samestate_lr1e-4_plus2000steps.npz'), [('B', B), ('C', C)])
