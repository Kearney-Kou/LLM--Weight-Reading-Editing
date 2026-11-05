"""运行方式：在 3_评估_勿给编辑者 根目录下执行 python3 代码/pack_heldout_C.py <HW1P2 数据根目录>
测试集 C：从 train-clean-100 中抽 1500 句，排除训练用的 12000 句（种子 2026）和放弃的 3000 句（种子 11785），用种子 314159 抽取。处理方式与其他数据相同。"""
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
