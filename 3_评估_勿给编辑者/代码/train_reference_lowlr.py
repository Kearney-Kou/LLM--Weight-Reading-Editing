"""How to run: from the 3_评估_勿给编辑者 root, run python3 代码/train_reference_lowlr.py
Extra reference: starting from the subject network, switch to learning rate 1e-4 (new AdamW state), continue training 2000 steps, save reference_lr1e-4_plus2000steps.npz.
Purpose: estimate how much "potential" this network has (at constant lr 1e-3 the weights bounce back and forth at the bottom; lowering the learning rate quickly captures part of the potential)."""
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
opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01); lossf = torch.nn.CrossEntropyLoss()
perm = torch.randperm(len(tr))
for s in range(2000):
    idx = perm[s * 1024:(s + 1) * 1024]
    loss = lossf(net(tr.windows(idx)), tr.y[idx]); opt.zero_grad(); loss.backward(); opt.step()
out = {f'W{i+1}': lin[i].weight.detach().T.numpy().copy() for i in range(3)} | {f'b{i+1}': lin[i].bias.detach().numpy().copy() for i in range(3)}
np.savez('网络/reference_lr1e-4_plus2000steps.npz', **out)
print('REF lr1e-4 +2000 steps, full dev A acc', round(accuracy(out, dA), 4), flush=True)
