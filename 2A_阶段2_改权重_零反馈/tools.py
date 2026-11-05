"""编辑者用的工具（阶段 2）。用法：
    from tools import *
    w   = load_net('subject_net.npz')          # dict: W1 (2268,1024), b1, W2 (1024,1024), b2, W3 (1024,42), b3
    dA  = load_A()                             # 练习集 A：dA.y 是标签，dA.windows(idx) 给出 (len(idx), 2268) 的输入
    acts = forward(w, dA, idx)                 # 任意网络的中间激活：dict x / z1 / h1 / z2 / h2 / logits（不计次数）
    acc  = evaluate(w_edited, dA)              # 在 A 上测一个"修改后的网络"的准确率等：计入预算
    save_net(w_edited, 'edited_net.npz')       # 提交
网络：h1 = relu(x @ W1 + b1)；h2 = relu(h1 @ W2 + b2)；logits = h2 @ W3 + b3；预测 = argmax(logits)
输入下标 i = t*28 + f：t = 0..80 是时间（第 40 帧是被分类的中心帧，t<40 是过去，t>40 是未来），f = 0..27 是 MFCC 维度。
"""
import os, json, numpy as np   # 只依赖 numpy
HERE = os.path.dirname(os.path.abspath(__file__))
K = 40
NAMES = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
PHONEMES = ['[SIL]','AA','AE','AH','AO','AW','AY','B','CH','D','DH','EH','ER','EY','F','G','HH','IH','IY','JH','K','L','M','N','NG','OW','OY','P','R','S','SH','T','TH','UH','UW','V','W','Y','Z','ZH','[SOS]','[EOS]']
SHAPES = {'W1': (2268, 1024), 'b1': (1024,), 'W2': (1024, 1024), 'b2': (1024,), 'W3': (1024, 42), 'b3': (42,)}
BUDGET = 0   # 本条件的评估预算
COUNTER = os.path.join(HERE, 'eval_log.jsonl')

class Frames:
    def __init__(self, parts):
        ds = [np.load(p) for p in parts]
        x = np.concatenate([d['x'] for d in ds]).astype(np.float32)
        self.y = np.concatenate([d['y'] for d in ds]).astype(np.int64)
        self.lens = np.concatenate([d['lens'] for d in ds])
        L = self.lens
        ps = np.concatenate([[0], np.cumsum(L + 2 * K)[:-1]]); fs = np.concatenate([[0], np.cumsum(L)[:-1]])
        self.P = np.zeros((int((L + 2 * K).sum()), 28), np.float32); self.center = np.empty(int(L.sum()), np.int64)
        for u in range(len(L)):
            self.P[ps[u] + K: ps[u] + K + L[u]] = x[fs[u]: fs[u] + L[u]]
            self.center[fs[u]: fs[u] + L[u]] = ps[u] + K + np.arange(L[u])
        self.utt = np.repeat(np.arange(len(L)), L)   # 每帧属于第几句
        self.offs = np.arange(-K, K + 1)
    def __len__(self): return len(self.y)
    def windows(self, idx):
        idx = np.asarray(idx)
        return self.P[self.center[idx][:, None] + self.offs].reshape(len(idx), -1)

def load_A():
    return Frames([os.path.join(HERE, 'dev_A_part1.npz'), os.path.join(HERE, 'dev_A_part2.npz')])

def load_net(path):
    d = np.load(path if os.path.isabs(path) else os.path.join(HERE, path))
    return {k: d[k].astype(np.float32).copy() for k in NAMES}

def save_net(w, path='edited_net.npz'):
    for k in NAMES:
        a = np.asarray(w[k], np.float32)
        assert a.shape == SHAPES[k], (k, a.shape) ; assert np.isfinite(a).all(), k
    np.savez(path if os.path.isabs(path) else os.path.join(HERE, path), **{k: np.asarray(w[k], np.float32) for k in NAMES})

def forward(w, data, idx):
    """返回 numpy 数组：x / z1 / h1 / z2 / h2 / logits"""
    x = data.windows(idx); z1 = x @ w['W1'] + w['b1']; h1 = np.maximum(z1, 0)
    z2 = h1 @ w['W2'] + w['b2']; h2 = np.maximum(z2, 0); lg = h2 @ w['W3'] + w['b3']
    return {'x': x, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'logits': lg}

def predict(w, data, bs=8192):
    w = {k: np.asarray(w[k], np.float32) for k in NAMES}
    return np.concatenate([forward(w, data, np.arange(i, min(i + bs, len(data))))['logits'].argmax(1) for i in range(0, len(data), bs)])

def evaluate(w, data, note=''):
    """在 A 上测一个修改后的网络：返回准确率和每类召回率。每调用一次用掉 1 次预算（原网络 subject_net 也算，如果你用它调用）。"""
    used = sum(1 for _ in open(COUNTER)) if os.path.exists(COUNTER) else 0
    if used >= BUDGET:
        raise RuntimeError(f'评估预算已用完：{used}/{BUDGET}')
    p = predict(w, data); ok = (p == data.y)
    acc = float(ok.mean())
    rec = {PHONEMES[c]: round(float(ok[data.y == c].mean()), 4) for c in range(40) if (data.y == c).any()}
    with open(COUNTER, 'a') as f: f.write(json.dumps({'n': used + 1, 'acc': acc, 'note': note}, ensure_ascii=False) + '\n')
    print(f'[评估 {used + 1}/{BUDGET}] A 上准确率 {acc:.4f}  {note}')
    return {'acc': acc, 'recall': rec}
