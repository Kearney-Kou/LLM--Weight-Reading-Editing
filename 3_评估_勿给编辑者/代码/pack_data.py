# Pack the experiment data from the raw features (read-only): 3000 random training utterances + all development utterances
# Cepstral normalization per utterance (standardize by column); features stored as float16, labels as uint8 ([SOS]/[EOS] removed)
import os, sys, numpy as np
ROOT = sys.argv[1]
PHONEMES = ['[SIL]','AA','AE','AH','AO','AW','AY','B','CH','D','DH','EH','ER','EY','F','G','HH','IH','IY','JH','K','L','M','N','NG','OW','OY','P','R','S','SH','T','TH','UH','UW','V','W','Y','Z','ZH','[SOS]','[EOS]']
P = {p: i for i, p in enumerate(PHONEMES)}
def pack(part, n, seed, out):
    names = sorted(os.listdir(os.path.join(ROOT, part, 'mfcc')))
    if n is not None:
        rng = np.random.default_rng(seed)
        names = sorted(rng.choice(names, size=n, replace=False))
    X, Y, lens = [], [], []
    for f in names:
        m = np.load(os.path.join(ROOT, part, 'mfcc', f))
        m = (m - m.mean(0, keepdims=True)) / (m.std(0, keepdims=True) + 1e-8)
        t = np.load(os.path.join(ROOT, part, 'transcript', f))[1:-1]
        assert len(t) == len(m), f
        X.append(m.astype(np.float16)); Y.append(np.array([P[p] for p in t], np.uint8)); lens.append(len(m))
    np.savez(out, x=np.concatenate(X), y=np.concatenate(Y), lens=np.array(lens, np.int64), names=np.array(names))
    print(out, sum(lens), len(names))
pack('dev-clean', None, 0, '数据/原始打包/dev_all.npz')
pack('train-clean-100', 3000, 11785, '数据/原始打包/train_subset3000.npz')
