English usage notes for `tools.py` in this folder. The Chinese original is the authoritative record. `tools.py` itself is left byte-for-byte unchanged, because it is part of the experimental material: it is exactly the file the editor received. This note translates its docstrings, comments and messages; the code is unchanged.

---

# `tools.py`: the editor's tools (stage 2)

`tools.py` depends only on NumPy. It expects `subject_net.npz`, `dev_A_part1.npz` and `dev_A_part2.npz` in the same folder. The two data files are not included; the README section "Reproducing" shows how to rebuild them. The weights are in the release asset `weights.zip` (see `WEIGHTS.md`).

In this folder the evaluation budget is `BUDGET = 20` (with feedback): up to 20 calls to `evaluate` on A. The editor used 2; they are recorded in `eval_log.jsonl`.

## Module docstring (translated)

```text
Tools for the editor (stage 2). Usage:
    from tools import *
    w   = load_net('subject_net.npz')          # dict: W1 (2268,1024), b1, W2 (1024,1024), b2, W3 (1024,42), b3
    dA  = load_A()                             # practice set A: dA.y are the labels, dA.windows(idx) gives inputs of shape (len(idx), 2268)
    acts = forward(w, dA, idx)                 # intermediate activations of any network: dict x / z1 / h1 / z2 / h2 / logits (not counted)
    acc  = evaluate(w_edited, dA)              # measure the accuracy etc. of an "edited network" on A: counts against the budget
    save_net(w_edited, 'edited_net.npz')       # submit
Network: h1 = relu(x @ W1 + b1); h2 = relu(h1 @ W2 + b2); logits = h2 @ W3 + b3; prediction = argmax(logits)
Input index i = t*28 + f: t = 0..80 is time (frame 40 is the centre frame being classified, t<40 is the past, t>40 is the future), f = 0..27 is the MFCC dimension.
```

## Functions

| Name | What it does |
|---|---|
| `load_net(path)` | Loads `W1, b1, W2, b2, W3, b3` from an `.npz` file as float32 arrays (relative paths are resolved against this folder). |
| `load_A()` | Loads practice set A from `dev_A_part1.npz` and `dev_A_part2.npz` as a `Frames` object. |
| `Frames` | Holds the frames of set A: `y` (labels), `lens` (frames per utterance), `utt` (which utterance each frame belongs to); `windows(idx)` returns the 81-frame context windows, shape `(len(idx), 2268)`, padded with zero frames at utterance boundaries (`K = 40`). |
| `forward(w, data, idx)` | Returns the intermediate activations `x / z1 / h1 / z2 / h2 / logits` for the frames `idx`. Does not use the budget. |
| `predict(w, data, bs=8192)` | Returns `argmax(logits)` for all frames, in batches. Does not use the budget. |
| `evaluate(w, data, note='')` | Measures a network on A and returns `{'acc': accuracy, 'recall': recall of each class}` (classes 0–39 that occur in A, keyed by phoneme name). Each call uses one unit of budget and appends one line `{"n", "acc", "note"}` to `eval_log.jsonl` in this folder. |
| `save_net(w, path='edited_net.npz')` | Checks shapes and finiteness, then saves the six arrays as float32 (the submission). |

`PHONEMES` lists the 42 labels in index order: `[SIL]`, the 39 ARPAbet phonemes in alphabetical order, `[SOS]`, `[EOS]`.

## Comments and messages (translated)

| Code | Chinese text | English |
|---|---|---|
| `import os, json, numpy as np` | `# 只依赖 numpy` | `# depends only on numpy` |
| `BUDGET = 20` | `# 本条件的评估预算` | `# the evaluation budget for this condition` |
| `self.utt = np.repeat(...)` | `# 每帧属于第几句` | `# which utterance each frame belongs to` |
| `forward` docstring | `返回 numpy 数组：x / z1 / h1 / z2 / h2 / logits` | `Returns numpy arrays: x / z1 / h1 / z2 / h2 / logits` |
| `evaluate` docstring | `在 A 上测一个修改后的网络：返回准确率和每类召回率。每调用一次用掉 1 次预算（原网络 subject_net 也算，如果你用它调用）。` | `Measures an edited network on A: returns the accuracy and the recall of each class. Each call uses 1 unit of the budget (the original network subject_net also counts, if you call it with that).` |
| error raised by `evaluate` | `评估预算已用完：{used}/{BUDGET}` | `Evaluation budget used up: {used}/{BUDGET}` |
| message printed by `evaluate` | `[评估 {used + 1}/{BUDGET}] A 上准确率 {acc:.4f}  {note}` | `[evaluation {used + 1}/{BUDGET}] accuracy on A {acc:.4f}  {note}` |

## `eval_log.jsonl` (translated notes)

`eval_log.jsonl` is kept unchanged. Its two `note` fields were written by the editor:

| `n` | `note` (Chinese) | English |
|---|---|---|
| 1 | `E1: W1 f26 行置零 + b2 均值补偿` | `E1: zero the f26 rows of W1 + b2 mean compensation` |
| 2 | `E2: E1 + b3 一次性先验修正 log(pi/qbar)` | `E2: E1 + one-off prior correction of b3, log(pi/qbar)` |
