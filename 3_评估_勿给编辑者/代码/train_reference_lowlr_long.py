"""第四版补充实验（post hoc，设置和预测见 结果/预测_第四版补充实验.md）：lr 1e-4 训练更久。
与 train_reference_lowlr.py 完全相同（从被试网络出发、新 AdamW、lr 1e-4、wd 0.01、batch 1024、torch.manual_seed(0)、同一个 randperm），
只是训练到 10,000 步，并在第 2,000/4,000/6,000/8,000/10,000 步保存检查点 网络/reference_lr1e-4_long_step{N}.npz（不覆盖已有文件）。
用法（在 3_评估_勿给编辑者 根目录下）：python3 代码/train_reference_lowlr_long.py <chunk0> <chunk1> <chunk2>"""
import numpy as np, torch, sys, time, os
from wlib import Frames, accuracy
from train_subject_lib import TrainFrames
torch.manual_seed(0); torch.set_num_threads(2)
tr = TrainFrames(sys.argv[1:]); dA = Frames('数据/dev_A.npz')
w = np.load('网络/subject_net.npz')
net = torch.nn.Sequential(torch.nn.Linear(2268, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 42))
lin = [m for m in net if isinstance(m, torch.nn.Linear)]
with torch.no_grad():
    for i in range(3):
        lin[i].weight.copy_(torch.from_numpy(w[f'W{i+1}']).T); lin[i].bias.copy_(torch.from_numpy(w[f'b{i+1}']))
opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01); lossf = torch.nn.CrossEntropyLoss()
perm = torch.randperm(len(tr))
assert 10000 * 1024 <= len(perm)
t0 = time.time()
for s in range(10000):
    idx = perm[s * 1024:(s + 1) * 1024]
    loss = lossf(net(tr.windows(idx)), tr.y[idx]); opt.zero_grad(); loss.backward(); opt.step()
    if (s + 1) % 2000 == 0:
        out = {f'W{i+1}': lin[i].weight.detach().T.numpy().copy() for i in range(3)} | {f'b{i+1}': lin[i].bias.detach().numpy().copy() for i in range(3)}
        p = f'网络/reference_lr1e-4_long_step{s + 1}.npz'; assert not os.path.exists(p)
        np.savez(p, **out)
        print(f'step {s + 1}  full dev A acc {accuracy(out, dA):.4f}  {time.time() - t0:.0f}s', flush=True)
        if s + 1 == 2000:
            ref = np.load('网络/reference_lr1e-4_plus2000steps.npz')
            print('  与 reference_lr1e-4_plus2000steps.npz 的最大绝对差：', {k: float(np.abs(out[k] - ref[k]).max()) for k in out}, flush=True)
