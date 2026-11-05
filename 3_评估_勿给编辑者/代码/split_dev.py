import numpy as np, sys
d = np.load(sys.argv[1])
x, y, lens, names = d['x'], d['y'], d['lens'], d['names']
rng = np.random.default_rng(20261103)
perm = rng.permutation(len(lens))
A = np.sort(perm[: len(lens) // 2]); B = np.sort(perm[len(lens) // 2:])
starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
def sub(ids, out):
    xs = np.concatenate([x[starts[u]: starts[u] + lens[u]] for u in ids])
    ys = np.concatenate([y[starts[u]: starts[u] + lens[u]] for u in ids])
    np.savez(out, x=xs, y=ys, lens=lens[ids], names=names[ids])
    print(out, len(ids), len(ys))
sub(A, '数据/dev_A.npz'); sub(B, '数据/dev_B.npz')
