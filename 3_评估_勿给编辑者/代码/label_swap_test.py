"""审稿意见 3：音系相关 r 能否区分"逐个标签都对"和"音系相近的标签互换"？
用研究者自建的音系特征表（不是 1C 的那张），在原网络 W3 上复现 1C 的检验，再比较几类更难的替代映射。"""
import numpy as np, sys
W3 = np.load(sys.argv[1])['W3'].astype(np.float64)          # 1024 x 42
P = 'AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH'.split()
V = {'AA':'low back','AE':'low front','AH':'mid central','AO':'mid back round','AW':'low central diph','AY':'low central diph',
     'EH':'mid front','ER':'mid central rhotic','EY':'mid front diph','IH':'high front','IY':'high front tense','OW':'mid back round diph',
     'OY':'mid back round diph','UH':'high back round','UW':'high back round tense'}
C = {'B':'stop bilabial voiced','CH':'affricate postalveolar','D':'stop alveolar voiced','DH':'fricative dental voiced','F':'fricative labiodental',
     'G':'stop velar voiced','HH':'fricative glottal','JH':'affricate postalveolar voiced','K':'stop velar','L':'liquid alveolar voiced',
     'M':'nasal bilabial voiced','N':'nasal alveolar voiced','NG':'nasal velar voiced','P':'stop bilabial','R':'liquid postalveolar voiced',
     'S':'fricative alveolar','SH':'fricative postalveolar','T':'stop alveolar','TH':'fricative dental','V':'fricative labiodental voiced',
     'W':'glide bilabial voiced','Y':'glide palatal voiced','Z':'fricative alveolar voiced','ZH':'fricative postalveolar voiced'}
desc = {p: ('vowel voiced ' + V[p]) if p in V else ('consonant ' + C[p]) for p in P}
vocab = sorted({t for d in desc.values() for t in d.split()})
F = np.array([[t in desc[p].split() for t in vocab] for p in P], float)
Fn = F / np.linalg.norm(F, axis=1, keepdims=True); S_ph = Fn @ Fn.T
M = W3[:, 1:40]; M = M / np.linalg.norm(M, axis=0, keepdims=True); S_w = M.T @ M
iu = np.triu_indices(39, 1)
def r_of(perm):   # perm[i] = 第 i 个 W3 列被赋予的音素下标
    return np.corrcoef(S_w[iu], S_ph[np.ix_(perm, perm)][iu])[0, 1]
ident = np.arange(39); r0 = r_of(ident)
rng = np.random.default_rng(0)
rand = np.array([r_of(rng.permutation(39)) for _ in range(5000)])
shift = r_of(np.roll(ident, 1))
swaps = []
for i in range(39):
    for j in range(i + 1, 39):
        p = ident.copy(); p[[i, j]] = p[[j, i]]; swaps.append((r_of(p), P[i], P[j], S_ph[i, j]))
sw = np.array([s[0] for s in swaps]); simp = np.array([s[3] for s in swaps])
close = simp >= np.quantile(simp, 0.9)      # 音系最相近的 10% 的标签对
def within_class_shuffle(n):
    groups = {}
    for k, p in enumerate(P): groups.setdefault(desc[p].split()[1] if p in C else 'vowel', []).append(k)
    out = []
    for _ in range(n):
        perm = ident.copy()
        for g in groups.values(): perm[g] = rng.permutation(g)
        out.append(r_of(perm))
    return np.array(out), {k: [P[i] for i in v] for k, v in groups.items()}
wc, groups = within_class_shuffle(2000)
print(f'真实映射 r = {r0:.3f}')
print(f'随机排列（5000 次）：均值 {rand.mean():.3f}，sd {rand.std():.3f}，最大 {rand.max():.3f}，z = {(r0-rand.mean())/rand.std():.1f}')
print(f'整体错位一位：r = {shift:.3f}')
print(f'单对互换（全部 {len(sw)} 对）：r 范围 [{sw.min():.3f}, {sw.max():.3f}]；r 比真实映射高的有 {(sw > r0).sum()} 对；与真实映射相差 <0.01 的有 {(abs(sw-r0) < 0.01).sum()} 对')
print(f'音系最相近的 10% 标签对（{close.sum()} 对）互换后：r 范围 [{sw[close].min():.3f}, {sw[close].max():.3f}]，平均下降 {r0 - sw[close].mean():.4f}')
print('举例（清浊对）：', ', '.join(f'{a}/{b} {r:.3f}' for r, a, b, _ in swaps if {a, b} in [{'P','B'},{'T','D'},{'K','G'},{'S','Z'},{'F','V'},{'IY','IH'},{'UW','UH'}]))
print(f'按发音方式分组、组内随机打乱（2000 次；分组 {", ".join(f"{k}:{len(v)}" for k, v in groups.items())}）：r 均值 {wc.mean():.3f}，范围 [{wc.min():.3f}, {wc.max():.3f}]，≥ 真实映射的比例 {(wc >= r0).mean():.3f}')
