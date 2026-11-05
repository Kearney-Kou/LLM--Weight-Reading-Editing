"""The rules frozen in the pre-registration (H1, H2, H3, H2+cal), and the "delete only, no compensation" versions used for mechanism separation."""
import numpy as np, torch
from wlib import Frames
A = Frames('数据/dev_A.npz')
_idx = torch.from_numpy(np.sort(np.random.default_rng(0).choice(len(A), 40000, replace=False)))
XA = A.windows(_idx).numpy()
xc = A.P[A.center].numpy(); yA = A.y.numpy()
def h1mean(w): return np.maximum(XA @ w['W1'] + w['b1'], 0).mean(0)
def zero_comp(w, mask, comp=True):
    v = {k: x.copy() for k, x in w.items()}; v['W1'][mask] = 0
    if comp: v['b2'] = w['b2'] + (h1mean(w) - h1mean(v)) @ w['W2']
    return v
def rownorm_mask(w, n=18):
    rn = np.linalg.norm(w['W1'].reshape(81, 28, 1024), axis=2).mean(1); ts = np.argsort(rn)[:n]
    m = np.zeros((81, 28, 1024), bool); m[ts] = True; return m.reshape(2268, 1024), sorted((ts - 40).tolist())
def min_eta_feature():
    eta = []
    for f in range(28):
        v = xc[:, f]; c = np.bincount(yA, minlength=42); m = np.bincount(yA, v, 42) / np.maximum(c, 1)
        eta.append(float((c * (m - v.mean()) ** 2).sum() / ((v - v.mean()) ** 2).sum()))
    return int(np.argmin(eta))
def feat_mask(f):
    m = np.zeros((81, 28, 1024), bool); m[:, f] = True; return m.reshape(2268, 1024)
def H1(w): return zero_comp(w, rownorm_mask(w)[0])
def H2(w): return zero_comp(w, feat_mask(min_eta_feature()))
def H3(w): return H2(H1(w))
def calibrate(w):
    t = {k: torch.as_tensor(x) for k, x in w.items()}; q = np.zeros(42); N = len(A)
    with torch.no_grad():
        for i in range(0, N, 16384):
            x = A.windows(torch.arange(i, min(i + 16384, N)))
            lg = torch.relu(torch.relu(x @ t['W1'] + t['b1']) @ t['W2'] + t['b2']) @ t['W3'] + t['b3']
            q += torch.softmax(lg, 1).sum(0).numpy()
    q /= N; pi = np.bincount(yA, minlength=42) / N
    v = {k: x.copy() for k, x in w.items()}; v['b3'][:40] += np.log(pi[:40] / q[:40]).astype(np.float32); v['b3'][40:] = -20.0
    return v
def H2cal(w): return calibrate(H2(w))
