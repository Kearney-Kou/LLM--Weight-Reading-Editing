"""补齐 2×2 对照（事后补做，未预注册）：保留原 AdamW 状态、把学习率降到 1e-4，再训练 2000 步。
做法：用与 train_subject.py 完全相同的种子和流程重新训练到第 9400 步，先核对权重与 subject_net.npz 一致，
然后在同一个优化器上把 lr 改成 1e-4，沿原来的数据顺序继续 2000 步。
用法：python3 train_reference_samestate_lr1e-4.py <chunk0> <chunk1> <chunk2>"""
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
