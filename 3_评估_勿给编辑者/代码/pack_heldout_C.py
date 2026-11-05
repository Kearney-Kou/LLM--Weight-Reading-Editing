"""How to run: from the 3_评估_勿给编辑者 root, run python3 代码/pack_heldout_C.py <data root>
Test set C: draw 1500 utterances from the training data, excluding the 12000 used for training (seed 2026) and the abandoned 3000 (seed 11785), drawn with seed 314159. Processed the same way as the other data."""
import os, sys, numpy as np
ROOT = sys.argv[1]
PHONEMES = ['[SIL]','AA','AE','AH','AO','AW','AY','B','CH','D','DH','EH','ER','EY','F','G','HH','IH','IY','JH','K','L','M','N','NG','OW','OY','P','R','S','SH','T','TH','UH','UW','V','W','Y','Z','ZH','[SOS]','[EOS]']
P = {p: i for i, p in enumerate(PHONEMES)}
names = sorted(os.listdir(os.path.join(ROOT, 'train-clean-100', 'mfcc')))
used = set(np.random.default_rng(2026).choice(names, size=12000, replace=False)) | set(np.random.default_rng(11785).choice(names, size=3000, replace=False))
pool = sorted(n for n in names if n not in used)
sel = sorted(np.random.default_rng(314159).choice(pool, size=1500, replace=False))
X, Y, lens = [], [], []
for f in sel:
    m = np.load(os.path.join(ROOT, 'train-clean-100', 'mfcc', f))
    m = (m - m.mean(0, keepdims=True)) / (m.std(0, keepdims=True) + 1e-8)
    t = np.load(os.path.join(ROOT, 'train-clean-100', 'transcript', f))[1:-1]
    assert len(t) == len(m), f
    X.append(m.astype(np.float16)); Y.append(np.array([P[p] for p in t], np.uint8)); lens.append(len(m))
np.savez('数据/test_C.npz', x=np.concatenate(X), y=np.concatenate(Y), lens=np.array(lens, np.int64), names=np.array(sel))
print('pool', len(pool), 'C utts', len(sel), 'frames', sum(lens))
