"""额外参照（机制分离）：从被试网络出发，重置 AdamW 状态、学习率保持 1e-3，训练 2000 步。与"重置状态 + lr 1e-4"对比，分离"降学习率"和"重置优化器状态"两个因素。
目的：估计这个网络的"潜力"有多大（恒定 lr 1e-3 时权重在底部来回抖，降学习率能很快拿到一部分潜力）。"""
import numpy as np, torch, sys
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
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01); lossf = torch.nn.CrossEntropyLoss()
perm = torch.randperm(len(tr))
for s in range(2000):
    idx = perm[s * 1024:(s + 1) * 1024]
    loss = lossf(net(tr.windows(idx)), tr.y[idx]); opt.zero_grad(); loss.backward(); opt.step()
out = {f'W{i+1}': lin[i].weight.detach().T.numpy().copy() for i in range(3)} | {f'b{i+1}': lin[i].bias.detach().numpy().copy() for i in range(3)}
np.savez('网络/reference_fresh_lr1e-3_plus2000steps.npz', **out)
print('REF fresh lr1e-3 +2000 steps, full dev A acc', round(accuracy(out, dA), 4), flush=True)
