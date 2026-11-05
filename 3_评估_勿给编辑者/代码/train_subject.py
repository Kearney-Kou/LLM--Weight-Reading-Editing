"""How to run: from the 3_评估_勿给编辑者 root, run python3 代码/train_subject.py
Train the subject network: 2268->1024->1024->42, ReLU, no BN/Dropout, AdamW lr 1e-3 (constant), weight decay 0.01, batch 1024.
Training data: 12000 random utterances of the training data (seed 2026).
Every 200 steps, measure once on a fixed 100,000 frames of dev A; the first time it is >= 70%, save subject_net.npz (the subject network).
Then continue training from the same state, and save a reference network reference_plus<N>steps.npz after 500 / 1000 / 2000 / 4000 further steps."""
import numpy as np, torch, time, sys
from wlib import Frames, accuracy
torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(2)
K = 40

class TrainFrames:   # training data stored as float16 to save memory
    def __init__(self, paths):
        ds = [np.load(p) for p in paths]
        x = np.concatenate([d['x'] for d in ds]); self.y = torch.from_numpy(np.concatenate([d['y'] for d in ds]).astype(np.int64))
        L = np.concatenate([d['lens'] for d in ds])
        ps = np.concatenate([[0], np.cumsum(L + 2 * K)[:-1]]); fs = np.concatenate([[0], np.cumsum(L)[:-1]])
        P = np.zeros((int((L + 2 * K).sum()), 28), np.float16); center = np.empty(int(L.sum()), np.int64)
        for u in range(len(L)):
            P[ps[u] + K: ps[u] + K + L[u]] = x[fs[u]: fs[u] + L[u]]
            center[fs[u]: fs[u] + L[u]] = ps[u] + K + np.arange(L[u])
        del x
        self.P = torch.from_numpy(P); self.center = torch.from_numpy(center); self.offs = torch.arange(-K, K + 1)
    def __len__(self): return len(self.y)
    def windows(self, idx):
        return self.P[self.center[idx][:, None] + self.offs].reshape(len(idx), -1).float()

tr = TrainFrames(sys.argv[1:]); dA = Frames('数据/dev_A.npz')
print('train frames', len(tr), flush=True)
mon = torch.from_numpy(np.random.default_rng(1).choice(len(dA), 100000, replace=False))
net = torch.nn.Sequential(torch.nn.Linear(2268, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 42))
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
lossf = torch.nn.CrossEntropyLoss()
def export():
    l = [m for m in net if isinstance(m, torch.nn.Linear)]
    return {f'W{i+1}': l[i].weight.detach().T.numpy().copy() for i in range(3)} | {f'b{i+1}': l[i].bias.detach().numpy().copy() for i in range(3)}
@torch.no_grad()
def mon_acc():
    return (net(dA.windows(mon)).argmax(1) == dA.y[mon]).float().mean().item()
BS = 1024; step = 0; t0 = time.time(); phase = 'to70'; after = 0; REFS = [500, 1000, 2000, 4000]
while True:
    perm = torch.randperm(len(tr))
    for i in range(0, len(tr) - BS + 1, BS):
        idx = perm[i:i + BS]
        loss = lossf(net(tr.windows(idx)), tr.y[idx])
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
        if phase == 'to70' and step % 200 == 0:
            a = mon_acc(); print(f'step {step} loss {loss.item():.3f} monA {a:.4f} {time.time()-t0:.0f}s', flush=True)
            if a >= 0.70:
                w = export(); np.savez('网络/subject_net.npz', **w)
                print('SUBJECT saved at step', step, 'frames seen', step * BS, 'full dev A acc', round(accuracy(w, dA), 4), flush=True)
                phase = 'refs'; continue
        if phase == 'refs':
            after += 1
            if after in REFS:
                w = export(); np.savez(f'网络/reference_plus{after}steps.npz', **w)
                print(f'REF +{after} steps saved, full dev A acc', round(accuracy(w, dA), 4), f'{time.time()-t0:.0f}s', flush=True)
                if after == REFS[-1]:
                    print('DONE', flush=True); sys.exit(0)
