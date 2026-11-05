"""Non-LLM baseline for reading the weights: using only the permuted W1 (net_q7), recover the 81×28 layout with a fixed algorithm and compare it with the true permutation.
Algorithm R1 (the same procedure as described by 1B, but completely fixed, without an LLM): row cosine similarity → graph of mutual top-2 nearest neighbours → connected components; path components of length exactly 81 are output in path order.
Algorithm R2 (clustering that does not rely on graph structure): hierarchical clustering (average linkage) of the similarities into 28 groups, ordered within each group by spectral ordering (Fiedler vector).
Scoring: a recovered channel is "correct" = all 81 rows come from the same original feature, and the order is monotone with the true time (forward or backward)."""
import numpy as np, time
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse.csgraph import connected_components
t0 = time.time()
perm = np.random.default_rng(777).permutation(2268)          # row i of q7 = original input perm[i]
W = np.load('../1_阶段1_读权重/net_q7.npz')['W1']
Wn = W / np.linalg.norm(W, axis=1, keepdims=True); S = Wn @ Wn.T; np.fill_diagonal(S, -1)
def score(chains):
    ok = 0
    for c in chains:
        o = perm[np.array(c)]; t, f = o // 28, o % 28
        if len(c) == 81 and len(set(f)) == 1 and (np.all(np.diff(t) == 1) or np.all(np.diff(t) == -1)): ok += 1
    return ok
# R1
top2 = np.argsort(-S, axis=1)[:, :2]
A = np.zeros((2268, 2268), bool)
for i in range(2268):
    for j in top2[i]:
        if i in top2[j]: A[i, j] = A[j, i] = True
n, lab = connected_components(A, directed=False)
chains = []; sizes = []
for k in range(n):
    nodes = np.where(lab == k)[0]; sizes.append(len(nodes))
    deg = A[nodes][:, nodes].sum(1)
    if len(nodes) == 81 and (deg <= 2).all() and (deg == 1).sum() == 2:
        start = nodes[np.where(deg == 1)[0][0]]; path = [start]; prev = -1; cur = start
        while True:
            nb = [j for j in np.where(A[cur])[0] if j != prev]
            if not nb: break
            prev, cur = cur, nb[0]; path.append(cur)
        chains.append(path)
sz = np.bincount(sizes)
print(f'R1 互近邻图：{n} 个分量，长度恰为 81 的路径 {len(chains)} 条，其中完全正确 {score(chains)} 条；分量大小分布（大小:个数）', {s: int(c) for s, c in enumerate(sz) if c and s >= 10})
# R2
D = 1 - S; np.fill_diagonal(D, 0); iu = np.triu_indices(2268, 1)
Z = linkage(D[iu], method='average'); g = fcluster(Z, 28, criterion='maxclust')
chains2 = []; pure = 0
for k in np.unique(g):
    nodes = np.where(g == k)[0]
    f = perm[nodes] % 28
    if len(set(f)) == 1 and len(nodes) == 81: pure += 1
    if len(nodes) > 2:
        Ssub = np.clip(S[np.ix_(nodes, nodes)], 0, None); np.fill_diagonal(Ssub, 0)
        Lap = np.diag(Ssub.sum(1)) - Ssub; v = np.linalg.eigh(Lap)[1][:, 1]
        chains2.append(list(nodes[np.argsort(v)]))
print(f'R2 层次聚类+谱排序：28 组中"81 行全属同一特征"的 {pure} 组，完全正确（含顺序）{score(chains2)} 条')
print(f'耗时 {time.time()-t0:.1f}s（2 核 CPU）；1B（LLM）的结果：25/28 完全正确')
