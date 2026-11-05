"""第三版补充实验 B1（post hoc，设置和预测见 结果/预测_第三版补充实验.md）：更强的"不知道答案"的通用读权重基线。
输入、打乱顺序和主打分与 reading_baseline.py（R1/R2）完全相同。在 3_评估_勿给编辑者 根目录下运行：
    python3 代码/reading_baseline_generic.py > 结果/reading_baseline_generic_out.txt
G1：kNN 图（余弦，取并集对称化，边权 max(S,0)）→ 谱聚类成 28 组 → 组内用 R2 的 Fiedler 排序。k ∈ {5,10,20,40} × 分配 {kmeans, cluster_qr}。
G2：整张图的谱排序（Fiedler 向量排成一条序列，切成 28 段 × 81）。图 ∈ {稠密 max(S,0), kNN k=5,10,20,40}。
G3a：Le Roux et al. 2007 照搬：Pearson 相关 ρ → D = sqrt(−log|ρ|) → Isomap(k) 嵌入 2 维 → 对齐到 28×81 网格（两种朝向）。k ∈ {4,8,16}。
G3b：Le Roux 适配到多条一维链：同样的距离和 kNN 图 → 测地距离 → 平均链接聚成 28 组 → 组内 1 维经典 MDS 排序。k ∈ {4,8,16}。
主指标与 R1/R2 相同（81 行同一特征且时间严格单调）；次要指标：纯组数、ARI、纯组内 |Spearman| 中位数。"""
import numpy as np, time, warnings
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import spearmanr
from sklearn.cluster import SpectralClustering
from sklearn.manifold import Isomap
from sklearn.metrics import adjusted_rand_score
warnings.filterwarnings('ignore')

perm = np.random.default_rng(777).permutation(2268)          # q7 第 i 行 = 原输入 perm[i]
W = np.load('../1_阶段1_读权重/net_q7.npz')['W1'].astype(np.float64)
Wn = W / np.linalg.norm(W, axis=1, keepdims=True); S = Wn @ Wn.T; np.fill_diagonal(S, -1)
TF, TT = perm % 28, perm // 28                                # 每行真实的特征、时间（只用于打分）

def score(chains):   # 与 reading_baseline.py 完全相同
    ok = 0
    for c in chains:
        o = perm[np.array(c)]; t, f = o // 28, o % 28
        if len(c) == 81 and len(set(f)) == 1 and (np.all(np.diff(t) == 1) or np.all(np.diff(t) == -1)): ok += 1
    return ok

def secondary(chains):
    lab = np.full(2268, -1)
    for g, c in enumerate(chains): lab[np.array(c)] = g
    pure = [c for c in chains if len(c) == 81 and len(set(TF[np.array(c)])) == 1]
    rho = [abs(spearmanr(np.arange(81), TT[np.array(c)])[0]) for c in pure]
    return len(pure), adjusted_rand_score(TF, lab), (np.median(rho) if rho else float('nan'))

def fiedler_order(nodes, A):   # R2 的组内排序：组内稠密 max(S,0) 的未归一化拉普拉斯
    if len(nodes) <= 2: return list(nodes)
    Ssub = A[np.ix_(nodes, nodes)].copy(); np.fill_diagonal(Ssub, 0)
    Lap = np.diag(Ssub.sum(1)) - Ssub; v = np.linalg.eigh(Lap)[1][:, 1]
    return list(np.array(nodes)[np.argsort(v)])

Sp = np.clip(S, 0, None); np.fill_diagonal(Sp, 0)
order_all = np.argsort(-S, axis=1)
def knn_graph(k, Sim=Sp, nbr=order_all):
    A = np.zeros((2268, 2268))
    for i in range(2268):
        A[i, nbr[i, :k]] = Sim[i, nbr[i, :k]]
    return np.maximum(A, A.T)

rows = []
def report(name, chains, t):
    sc = score(chains); pure, ari, rho = secondary(chains)
    sizes = sorted([len(c) for c in chains], reverse=True)
    rows.append((name, sc, pure, ari, rho, t))
    print(f'{name:38s} 正确 {sc:2d}/28  纯组 {pure:2d}  ARI {ari:.3f}  纯组|Spearman|中位 {rho:.3f}  耗时 {t:5.1f}s  组大小(前5/后3) {sizes[:5]}/{sizes[-3:]}', flush=True)

# ---------- G1 ----------
for k in [5, 10, 20, 40]:
    A = knn_graph(k)
    ncc = connected_components(csr_matrix(A > 0), directed=False)[0]
    for al in ['kmeans', 'cluster_qr']:
        t0 = time.time()
        kw = dict(n_init=10) if al == 'kmeans' else {}
        g = SpectralClustering(n_clusters=28, affinity='precomputed', assign_labels=al, random_state=0, **kw).fit_predict(A)
        chains = [fiedler_order(np.where(g == c)[0], Sp) for c in range(28)]
        report(f'G1 kNN k={k:2d}（{ncc} 个连通分量）+谱聚类[{al}]', chains, time.time() - t0)

# ---------- G2 ----------
for nm, A in [('稠密 max(S,0)', Sp)] + [(f'kNN k={k}', knn_graph(k)) for k in [5, 10, 20, 40]]:
    t0 = time.time()
    Lap = np.diag(A.sum(1)) - A; v = np.linalg.eigh(Lap)[1][:, 1]
    o = np.argsort(v, kind='stable'); chains = [list(o[81 * i: 81 * (i + 1)]) for i in range(28)]
    report(f'G2 全局谱排序，图 = {nm}', chains, time.time() - t0)

# ---------- G3（Le Roux et al. 2007）----------
Wc = W - W.mean(1, keepdims=True); Wc /= np.linalg.norm(Wc, axis=1, keepdims=True)
R = Wc @ Wc.T; np.fill_diagonal(R, 1.0)
D = np.sqrt(-np.log(np.clip(np.abs(R), 1e-12, 1.0))); np.fill_diagonal(D, 0.0)
for k in [4, 8, 16]:
    t0 = time.time()
    Y = Isomap(n_neighbors=k, n_components=2, metric='precomputed').fit_transform(D)
    t1 = time.time() - t0
    for ax in [0, 1]:
        o = np.argsort(Y[:, ax], kind='stable'); chains = []
        for i in range(28):
            seg = o[81 * i: 81 * (i + 1)]; chains.append(list(seg[np.argsort(Y[seg, 1 - ax], kind='stable')]))
        report(f'G3a Le Roux 照搬 Isomap2D k={k:2d}，按第{ax + 1}坐标分段', chains, t1)
for k in [4, 8, 16]:
    t0 = time.time()
    nbr = np.argsort(D, axis=1)[:, 1:k + 1]; G = np.zeros((2268, 2268))
    for i in range(2268): G[i, nbr[i]] = D[i, nbr[i]]
    G = np.maximum(G, G.T)
    ncc = connected_components(csr_matrix(G > 0), directed=False)[0]
    Gd = shortest_path(csr_matrix(G), directed=False)
    fin = np.isfinite(Gd); Gd[~fin] = 2 * Gd[fin].max()
    iu = np.triu_indices(2268, 1); g = fcluster(linkage(Gd[iu], method='average'), 28, criterion='maxclust')
    chains = []
    for c in np.unique(g):
        nodes = np.where(g == c)[0]
        if len(nodes) <= 2: chains.append(list(nodes)); continue
        Dg = Gd[np.ix_(nodes, nodes)] ** 2; n = len(nodes); J = np.eye(n) - 1 / n
        v = np.linalg.eigh(-0.5 * J @ Dg @ J)[1][:, -1]               # 1 维经典 MDS
        chains.append(list(nodes[np.argsort(v, kind='stable')]))
    report(f'G3b Le Roux 适配 测地+聚类 k={k:2d}（{ncc} 个连通分量）', chains, time.time() - t0)

print('\n汇总（22 个设置，全部报告）：')
for g in ['G1', 'G2', 'G3a', 'G3b']:
    r = [x for x in rows if x[0].startswith(g + ' ')]
    print(f'  {g}: 正确条数 {[x[1] for x in r]}，最好 {max(x[1] for x in r)}/28，最差 {min(x[1] for x in r)}/28')
print('对照：R1（1B 描述的流程）25/28；R2（平均链接层次聚类）0/28；1B（LLM）25/28')

# ---------- 追加的探索性设置（看过上面结果后才加，不在预设网格里；k=2 用到了 1B 的"两强"观察，不算通用方法）----------
print('\n诊断：第 r 近邻中"同特征且 |Δt|≤2"的比例')
for r in range(6):
    j = order_all[:, r]; print(f'  第{r + 1}近邻 {((TF[j] == TF) & (abs(TT[j] - TT) <= 2)).mean():.3f}（同特征 {(TF[j] == TF).mean():.3f}）')
W0 = np.empty_like(Wn); W0[perm] = Wn; W0 = W0.reshape(81, 28, 1024)   # 按真实顺序排回（只用于诊断）
print('同一特征内、相隔 L 帧的两行的余弦相似度（对 27 个特征（去掉 26）和所有偏移取平均 / 范围）：')
for Lg in [1, 2, 3, 5, 10, 20]:
    v = np.array([[(W0[t, f] * W0[t + Lg, f]).sum() for t in range(81 - Lg)] for f in range(28) if f != 26])
    print(f'  L={Lg:2d}: 平均 {v.mean():+.3f}，按偏移平均后的范围 [{v.mean(0).min():+.3f}, {v.mean(0).max():+.3f}]')
print('\n探索性追加：G1 的 k ∈ {2,3,4}')
rows_pre = len(rows)
for k in [2, 3, 4]:
    A = knn_graph(k)
    ncc = connected_components(csr_matrix(A > 0), directed=False)[0]
    for al in ['kmeans', 'cluster_qr']:
        t0 = time.time()
        kw = dict(n_init=10) if al == 'kmeans' else {}
        g = SpectralClustering(n_clusters=28, affinity='precomputed', assign_labels=al, random_state=0, **kw).fit_predict(A)
        chains = [fiedler_order(np.where(g == c)[0], Sp) for c in range(28)]
        report(f'G1* kNN k={k:2d}（{ncc} 个连通分量）+谱聚类[{al}]', chains, time.time() - t0)
