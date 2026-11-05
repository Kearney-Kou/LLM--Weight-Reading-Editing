# 训练数据扩大到 12000 句（随机抽，种子 2026），分 3 块打包，每块 4000 句；用法：python3 pack_train_chunks.py ROOT 块号(0/1/2)
import os, sys, numpy as np
ROOT, k = sys.argv[1], int(sys.argv[2])
PHONEMES = ['[SIL]','AA','AE','AH','AO','AW','AY','B','CH','D','DH','EH','ER','EY','F','G','HH','IH','IY','JH','K','L','M','N','NG','OW','OY','P','R','S','SH','T','TH','UH','UW','V','W','Y','Z','ZH','[SOS]','[EOS]']
P = {p: i for i, p in enumerate(PHONEMES)}
names = sorted(os.listdir(os.path.join(ROOT, 'train-clean-100', 'mfcc')))
names = sorted(np.random.default_rng(2026).choice(names, size=12000, replace=False))[k*4000:(k+1)*4000]
X, Y, lens = [], [], []
for f in names:
    m = np.load(os.path.join(ROOT, 'train-clean-100', 'mfcc', f))
    m = (m - m.mean(0, keepdims=True)) / (m.std(0, keepdims=True) + 1e-8)
    t = np.load(os.path.join(ROOT, 'train-clean-100', 'transcript', f))[1:-1]
    assert len(t) == len(m), f
    X.append(m.astype(np.float16)); Y.append(np.array([P[p] for p in t], np.uint8)); lens.append(len(m))
np.savez(f'数据/原始打包/train12000_chunk{k}.npz', x=np.concatenate(X), y=np.concatenate(Y), lens=np.array(lens, np.int64), names=np.array(names))
print(k, sum(lens))
