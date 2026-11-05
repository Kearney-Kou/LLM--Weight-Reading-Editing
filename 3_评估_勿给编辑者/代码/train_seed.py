"""新种子被试网络：与 train_subject.py 完全相同（数据、结构、优化器、监测子集、停止规则），只改随机种子；只保存被试网络。
用法：python3 train_seed.py <seed> <train chunk0> <chunk1> <chunk2>"""
import numpy as np, torch, time, sys
from wlib import Frames, accuracy
from train_subject_lib import TrainFrames
s = int(sys.argv[1]); torch.manual_seed(s); np.random.seed(s); torch.set_num_threads(2)
tr = TrainFrames(sys.argv[2:]); dA = Frames('数据/dev_A.npz')
mon = torch.from_numpy(np.random.default_rng(1).choice(len(dA), 100000, replace=False))   # 与原网络相同的监测子集
net = torch.nn.Sequential(torch.nn.Linear(2268, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 42))
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01); lossf = torch.nn.CrossEntropyLoss()
def export():
    l = [m for m in net if isinstance(m, torch.nn.Linear)]
    return {f'W{i+1}': l[i].weight.detach().T.numpy().copy() for i in range(3)} | {f'b{i+1}': l[i].bias.detach().numpy().copy() for i in range(3)}
BS = 1024; step = 0; t0 = time.time()
while True:
    perm = torch.randperm(len(tr))
    for i in range(0, len(tr) - BS + 1, BS):
        idx = perm[i:i + BS]; loss = lossf(net(tr.windows(idx)), tr.y[idx]); opt.zero_grad(); loss.backward(); opt.step(); step += 1
        if step % 200 == 0:
            with torch.no_grad(): a = (net(dA.windows(mon)).argmax(1) == dA.y[mon]).float().mean().item()
            print(f'seed {s} step {step} monA {a:.4f} {time.time()-t0:.0f}s', flush=True)
            if a >= 0.70:
                w = export(); np.savez(f'网络/subject_seed{s}.npz', **w)
                print(f'SUBJECT seed {s} saved at step {step}, full A acc {accuracy(w, dA):.4f}', flush=True); sys.exit(0)
