"""Completing the 2×2 control (done afterwards, not pre-registered): keep the original AdamW state, lower the learning rate to 1e-4, and train 2000 more steps.
Method: retrain to step 9400 with exactly the same seed and procedure as train_subject.py, first check that the weights match subject_net.npz,
then change lr to 1e-4 on the same optimizer and continue 2000 steps along the original data order.
Usage: python3 train_reference_samestate_lr1e-4.py <chunk0> <chunk1> <chunk2>"""
import numpy as np, torch, time, sys, copy
from wlib import Frames, accuracy
from train_subject_lib import TrainFrames
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2)
tr = TrainFrames(sys.argv[1:]); dA = Frames('数据/dev_A.npz')
mon = torch.from_numpy(np.random.default_rng(1).choice(len(dA), 100000, replace=False))
net = torch.nn.Sequential(torch.nn.Linear(2268, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 42))
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01); lossf = torch.nn.CrossEntropyLoss()
def export():
    l = [m for m in net if isinstance(m, torch.nn.Linear)]
    return {f'W{i+1}': l[i].weight.detach().T.numpy().copy() for i in range(3)} | {f'b{i+1}': l[i].bias.detach().numpy().copy() for i in range(3)}
def maxdiff(w, path): r = np.load(path); return max(float(np.abs(w[k] - r[k]).max()) for k in w)
BS = 1024; step = 0; t0 = time.time(); phase = 'to70'; after = 0
while True:
    perm = torch.randperm(len(tr))
    for i in range(0, len(tr) - BS + 1, BS):
        idx = perm[i:i + BS]; loss = lossf(net(tr.windows(idx)), tr.y[idx]); opt.zero_grad(); loss.backward(); opt.step(); step += 1
        if phase == 'to70' and step % 200 == 0:
            with torch.no_grad(): a = (net(dA.windows(mon)).argmax(1) == dA.y[mon]).float().mean().item()
            print(f'step {step} monA {a:.4f} {time.time()-t0:.0f}s', flush=True)
            if a >= 0.70:
                print('reached 70% at step', step, '; max |diff| vs subject_net.npz =', maxdiff(export(), '网络/subject_net.npz'), flush=True)
                for g in opt.param_groups: g['lr'] = 1e-4
                phase = 'refs'; continue
        if phase == 'refs':
            after += 1
            if after == 2000:
                w = export(); np.savez('网络/reference_samestate_lr1e-4_plus2000steps.npz', **w)
                print('REF same-state lr1e-4 +2000 saved; full dev A acc', round(accuracy(w, dA), 4), f'{time.time()-t0:.0f}s', flush=True); sys.exit(0)
