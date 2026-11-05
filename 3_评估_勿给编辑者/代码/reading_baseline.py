"""读权重的非 LLM 基线：只用打乱后的 W1（net_q7），用固定算法恢复 81×28 布局，和真实打乱顺序比对。
算法 R1（与 1B 描述的流程相同，但完全固定、不经 LLM）：行余弦相似度 → 互为前 2 近邻建图 → 连通分量；长度恰为 81 的路径分量按路径顺序输出。
算法 R2（不依赖图结构的聚类）：对相似度做层次聚类（平均链接）分成 28 组，组内按谱排序（Fiedler 向量）排顺序。
评分：一条恢复出的通道"正确" = 81 行都来自同一个原特征，且顺序与真实时间单调一致（正向或反向）。"""
import numpy as np, time
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse.csgraph import connected_components
t0 = time.time()
perm = np.random.default_rng(777).permutation(2268)          # q7 第 i 行 = 原输入 perm[i]
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
