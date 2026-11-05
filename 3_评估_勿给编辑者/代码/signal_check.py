# Experimenter's self-check: does readable structure really exist in the weights (not shown to the subjects)
import numpy as np
w = np.load('网络/subject_net.npz'); W1 = w['W1']                     # (2268, 1024)
R = np.linalg.norm(W1, axis=1).reshape(81, 28)                    # weight norm of each input
prof = R.mean(1); prof /= prof.max()
print('按时间偏移的平均权重范数（t-40 = -40..40，每 5 帧取一个）:')
print(' '.join(f'{t-40:+d}:{prof[t]:.2f}' for t in range(0, 81, 5)))
C = np.corrcoef(W1)                                               # weight correlation between inputs
lag = {k: np.mean([C[i, i + k] for i in range(2268 - k)]) for k in [1, 14, 27, 28, 29, 56]}
print('相隔 k 个输入的平均权重相关:', {k: round(v, 3) for k, v in lag.items()})
b3 = w['b3']; print('输出 bias 最大的 5 类:', np.argsort(-b3)[:5], ' [SOS]/[EOS] bias:', b3[40:42].round(3), ' 中位数:', np.median(b3).round(3))
