# Reading and editing the weights of a small MLP with a language model

English | [中文](README.zh-CN.md)

This repository contains the code, prompts, model outputs, pre-registration and results of an experiment that tests two things: whether a language model (Claude Opus 5.5) can tell what a trained neural network does from its weights alone, and whether it can make the network more accurate by editing the weights directly, without any training.

## Background and purpose

In the blog post *Sparks of Weight-Space Programming* ([post](https://rikiyat.github.io/blog/weight-space-programming/), [logs](https://github.com/RikiyaT/lm_understand-logs)), Takehi et al. (2026) propose that a frontier language model can reason about what each part of a network computes and then write weights directly, instead of obtaining them by optimization. They report that language models identified what networks had been trained on, repaired a failing policy and wrote small networks from scratch. They also report a limit: on a model without an obvious failure, the edits barely changed the benchmark score. Their interpretation is that editing can remove an obstacle that hides an ability the network already has, but cannot add an ability it lacks.

Their examples are policies, small programs and language models. This experiment asks whether the same observations hold in an ordinary supervised task: frame-level phoneme classification with a multilayer perceptron (MLP). The task has dense labels, no exploration problem and no compact symbolic solution, so training is the standard and effective way to improve the network. The setting also makes measurement simple: the network can be scored on a test set that the language model never sees, and every edit can be compared with fixed rules that use no language model and with a little more training.

## Experimental design

### Subject network

The network whose weights are read and edited is called the subject network. It is an MLP with 2,268 inputs, two hidden layers of 1,024 ReLU units and 42 outputs (3,416,106 parameters), without normalization or dropout:

```
h1 = relu(x @ W1 + b1)      W1: 2268 x 1024
h2 = relu(h1 @ W2 + b2)     W2: 1024 x 1024
logits = h2 @ W3 + b3       W3: 1024 x 42
```

The data are recordings of read English speech, converted into acoustic features. Each 10 ms frame has 28 MFCC features, normalized within each utterance. To classify a frame, the network sees a window of 81 frames: the frame itself and 40 frames on each side, zero-padded at utterance boundaries and flattened as input index `t*28 + f` (t = 0…80, frame 40 is the one being classified). There are 40 labels, silence plus 39 ARPAbet phonemes. The output layer has 42 units because the label list also contains `[SOS]` and `[EOS]`, which never occur as frame labels.

The network was trained with AdamW at a constant learning rate of 1e-3 (weight decay 0.01, batch size 1,024) on 12,000 training utterances. The stopping rule, fixed in advance, was to stop the first time accuracy on a fixed 100,000-frame subset of split A reached 70%. This happened at step 9,400, after about 0.63 epochs. The network was stopped early on purpose. An undertrained network leaves room for improvement, so an edit has something to gain, and the gain can be compared directly with what further training gives.

### Stage 1: reading the weights

Three independent sessions, called readers, were asked what the input data, the input structure, the task and the 42 classes are. They were told only that the network is an MLP classifier trained with cross-entropy on a real-world dataset with standardized inputs.

- 1A received the architecture description and no weights. It shows what can be guessed from the shapes alone (2,268 inputs, 42 classes).
- 1B received the same weights as 1C, with the rows of `W1` reordered by a secret random permutation.
- 1C received the original weights.

The permutation is the central control. In the original weights, inputs that are neighbours in time sit at indices 28 apart, so the layout could be read off from positions. The permutation removes this positional information: the permuted network computes the same function on identically permuted inputs, and the file contains nothing about the permutation. Any layout that 1B reconstructs must therefore come from the weight values. The scoring rubric was written before any answer was seen ([`结果/阶段1_答案.md`](3_评估_勿给编辑者/结果/阶段1_答案.md)).

### Stage 2: editing the weights

Two independent sessions, called editors, received the subject network, split A with labels, and a NumPy tool file (forward pass, activations, budgeted evaluation, saving). They were asked to make the network as accurate as possible on the hidden split B by editing its weights. Training of any kind was forbidden, and so was solving for weights by minimizing a loss on A, including closed-form fits and automated search. Each editor returned the edited network, an edit report with a diagnosis and a predicted accuracy on B, and a log of all code it ran.

- 2A had an evaluation budget of 0: it could not measure the accuracy of any edited network.
- 2B could evaluate up to 20 edited networks on A. It used 2.

### Information available to each condition

Unlike the readers, the editors were told the task, the features, the input layout, the label names and how the network was trained: the optimizer and its settings, the stopping step, and that the network is undertrained. The fixed rules H1, H2 and H2+cal in the table are described under Results.

| Condition | Weights | Told the task and training details | Data | Labels | Could measure the accuracy of edited networks | Language model |
|---|---|---|---|---|---|---|
| 1A | none (architecture only) | no | none | none | — | yes |
| 1B | permuted | no | none | none | — | yes |
| 1C | original | no | none | none | — | yes |
| 2A | original | yes | A | A | no (budget 0) | yes |
| 2B | original | yes | A | A | up to 20 evaluations on A | yes |
| H1 | original | — | 40,000 unlabelled frames of A | none | no | no |
| H2 | original | — | A | A | no | no |
| H2+cal | original | — | A | A | no | no |

All readers and editors were Claude Opus 5.5, each in a new session of the Claude desktop app with default settings. The app does not expose the model snapshot or the sampling temperature, so neither was fixed or recorded. 1B, 1C, 2A and 2B could run Python. 1A was asked for a one-shot answer and was not told that it could run code. The full prompts and task descriptions are in the stage folders, in the original Chinese and in English translation (`*.en.md`).

### Data splits

| Split | Source | Size | Use |
|---|---|---|---|
| Training | 12,000 utterances sampled at random from the training data | 15,203,231 frames | Training the subject network, the two additional seeds and the training references |
| A | a random half of the development data, split by utterance | 1,351 utterances, 947,349 frames | Stopping rule (100,000-frame subset); practice set for the editors; statistics used by the fixed rules |
| B | the other half of the development data | 1,352 utterances, 980,855 frames, 40 speakers | Hidden test set; every network is evaluated on it once |
| C | 1,500 further utterances from the training data, none used in training | 1,914,753 frames, 251 speakers | Test set for the pre-registered confirmatory test |

A and B share their 40 speakers, and no training speaker appears in either. The utterances in C are new, but its speakers also appear in the training set.

## Procedure

The experiment was carried out in this order.

1. Data. The features were packed (`pack_data.py`, `pack_train_chunks.py`), and the development data were split into A and B (`split_dev.py`).
2. Subject network. Two earlier configurations, trained on 3,000 utterances, were abandoned because accuracy levelled off around 65%, below the stopping threshold. Their logs are kept. The final configuration, trained on 12,000 utterances, met the stopping rule at step 9,400. Its accuracy is 69.80% on A, 70.05% on B and 71.10% on C.
3. Training references. To measure what ordinary training gains, training continued from the subject network with the same optimizer state for 500, 1,000, 2,000 and 4,000 further steps, and, separately, for 2,000 steps with a new AdamW state at a learning rate of 1e-4. Both were made before any editing.
4. Stage 1. The scoring rubric was written. Then 1A, 1B and 1C were run in separate sessions.
5. Stage 2. 2A and 2B were run in separate sessions. Each returned its edited network, edit report and code log.
6. Evaluation on B. Each submitted network was evaluated once on B. Confidence intervals come from a paired bootstrap over utterances, because frames within an utterance are strongly correlated. The gains were compared with null distributions (random edits of the same kind and size) and with simple rules that use no language model: deleting the 18 time offsets with the smallest weight norms, magnitude pruning of as many weights as 2A changed, and deleting the least informative input feature. These rules were designed after the results on B were known.
7. Pre-registered confirmatory test. Because the rules in step 6 were chosen after seeing B, they were frozen as rules H1, H2, H3 and H2+cal, together with four predictions P1–P4, in [`结果/预注册_确认性检验.md`](3_评估_勿给编辑者/结果/预注册_确认性检验.md) before any confirmatory run. Then the new test set C was built, and two new subject networks (seeds 1 and 2) were trained, identical to the subject network except for the random seed and never shown to a language model. The frozen rules were applied to all three networks and evaluated once on B and C, with bootstrap intervals over utterances and over speakers. In the same phase, two reading algorithms that use no language model (R1, R2) were run on the permuted weights, the label assignment proposed by 1C and the uninformative input feature were checked further, and two more training references (a new AdamW state at learning rate 1e-3; the same optimizer state at learning rate 1e-4) separated the effect of the learning rate from that of resetting the optimizer.
8. Supplementary analyses. These are not confirmatory tests, but for each of them the settings, predictions and decision rules were written down before it was run (`结果/预测_*.md`):
   - generic reading algorithms (G1–G3) that do not use knowledge of the answer;
   - H1 applied to networks trained further from the subject network;
   - training at a learning rate of 1e-4 for 10,000 steps, with H1 applied at every 2,000-step checkpoint;
   - a decomposition of how the far-context weights moved during that training;
   - H1 evaluated on training frames that the network had and had not been trained on.

## Results

Changes in accuracy are given in percentage points (pp) relative to the unedited network, with 95% confidence intervals from a paired bootstrap over utterances (2,000 resamples). All numbers below are taken from the files in [`3_评估_勿给编辑者/结果/`](3_评估_勿给编辑者/结果/); the source file is named under each table.

### Reading the weights

1A answered that the input is a sequence of hand keypoints (36 frames × 21 keypoints × 3 coordinates) for gesture or sign recognition, and did not mention speech. 1C inferred the 81 × 28 time-major layout, that the label belongs to the centre frame, that the task is frame-level phoneme classification, that class 0 is silence, that classes 1–39 are ARPAbet phonemes in alphabetical order, and that classes 40 and 41 almost never occur. Under the rubric, 1B reached the highest level with the silence bonus, and 1C reached the highest level with both bonus items. The answers are in [`1_阶段1_读权重/`](1_阶段1_读权重/).

The quantitative test is 1B: how many of the 28 input features it recovered as a complete chain of 81 frames in the correct order. The direction of a chain cannot be identified from the weights, so a reversed chain counts as correct.

| Method | Language model | Settings | Features recovered (of 28) |
|---|---|---|---|
| 1B | yes | 1 session | 25 |
| R1: graph of mutual nearest neighbours between rows of `W1` (the procedure 1B described, written as a fixed algorithm) | no | 1 | 25 (1.9 s on 2 CPU cores) |
| R2: average-linkage hierarchical clustering into 28 groups, ordered within each group by spectral ordering | no | 1 | 0 |
| G1: k-nearest-neighbour graph + spectral clustering | no | 8 | 0 in every setting |
| G2: global spectral ordering | no | 5 | 0 in every setting |
| G3a / G3b: the Isomap-based method of Le Roux et al. (2007), as published / adapted | no | 6 / 3 | 0 in every setting |

Sources: `reading_baseline_out.txt`, `reading_baseline_generic_out.txt`. Six further exploratory settings of G1 (k = 2, 3, 4) also recovered 0 features.

R1 shows that the procedure 1B chose is enough to recover the layout once it is written down. G1–G3 show that generic methods that do not contain this procedure do not find it. In this condition, the language model's contribution was the choice of analysis.

### Editing the weights

2A made five edits. The first two set to zero all `W1` weights at time offsets |Δt| = 30–38 (18 offsets, all 28 features) and then adjusted `b2` so that the mean input to the second layer over A is unchanged. This adjustment is called compensation below. The other three changed the output layer. 2B made two edits: it set to zero all `W1` weights of input feature 26, which it judged to be pure noise, with compensation, and it then shifted the output biases so that the network's average predicted class frequencies match the label frequencies of A.

The fixed rules repeat these edits without a language model. H1 sets to zero the 18 time offsets with the smallest mean `W1` row norm (this selects |Δt| = 29–37) and applies compensation. H2 sets to zero the input feature with the smallest share of between-class variance on A (feature 26) and applies compensation; it is identical to 2B's first edit. H2+cal adds 2B's output-bias correction to H2. H3 applies H1 and then H2.

| Network | Accuracy on B (%) | Change (pp) | 95% CI |
|---|---|---|---|
| Subject network, unedited | 70.05 | — | — |
| *Editors* | | | |
| 2A, as submitted (5 edits) | 70.44 | +0.39 | [+0.33, +0.45] |
| 2A, edits 1–2 only | 70.46 | +0.41 | [+0.35, +0.47] |
| 2B, as submitted (2 edits) | 70.26 | +0.21 | [+0.18, +0.24] |
| 2B, edit 1 only (= H2) | 70.14 | +0.09 | [+0.07, +0.12] |
| *Fixed rules* | | | |
| H1 | 70.45 | +0.40 | [+0.34, +0.46] |
| H2+cal | 70.26 | +0.21 | [+0.18, +0.24] |
| Magnitude pruning of as many `W1` weights as 2A changed, with compensation | 70.09 | +0.04 | [+0.02, +0.05] |
| Training references (starting from the subject network) | | | |
| +500 steps, same optimizer state, learning rate 1e-3 | 70.05 | −0.00 | [−0.12, +0.10] |
| +1,000 steps, same optimizer state, learning rate 1e-3 | 70.24 | +0.19 | [+0.07, +0.31] |
| +2,000 steps, same optimizer state, learning rate 1e-3 | 70.57 | +0.52 | [+0.40, +0.64] |
| +4,000 steps, same optimizer state, learning rate 1e-3 | 70.68 | +0.63 | [+0.51, +0.76] |
| +2,000 steps, new AdamW state, learning rate 1e-3 | 70.35 | +0.30 | [+0.18, +0.43] |
| +2,000 steps, same optimizer state, learning rate 1e-4 | 72.87 | +2.83 | [+2.74, +2.92] |
| +2,000 steps, new AdamW state, learning rate 1e-4 | 72.90 | +2.85 | [+2.76, +2.95] |

Sources: `final_table.txt`, `summary_out.txt` (H2+cal and the references with learning rate 1e-3 / new state and 1e-4 / same state).

- H1 reproduces 2A's gain, and H2+cal reproduces 2B's gain.
- Both edits are small compared with lowering the learning rate: 2,000 steps at 1e-4 gain about 2.8 pp, whether or not the optimizer state is reset. With the learning rate kept at 1e-3, a new optimizer state gives +0.30 pp.
- Edits of the same kind chosen at random lower accuracy. In 20 random choices of 18 time offsets, the change on B ranged from −0.97 to −26.96 pp; in 10 random choices of the same number of `W1` positions as 2A, from −8.58 to −9.83 pp. Deleting each input feature in turn, feature 26 gives the largest gain (+0.09 pp). Sources: `null_a.txt`, `null_b.txt`, `null_c.txt`.
- Each editor also predicted its accuracy on B: 2A about 70.0% (range 69.3–70.7%), 2B about 70.0% (range 69.6–70.4%). These predictions are in the edit reports `修改说明.md`.

### Pre-registered confirmatory test

The predictions, written before any confirmatory run:

- P1. On C, for the subject network, 2A's edit, 2A's edits 1–2, H1 and H3 each raise accuracy with a 95% interval that excludes 0, and H1 and 2A's edits 1–2 differ by at most 0.1 pp.
- P2. On C, H2 (identical to 2B's first edit) has a point estimate ≥ 0.
- P3. For seeds 1 and 2, H1 and H3 raise accuracy on both B and C with intervals that exclude 0, and H2 has point estimates ≥ 0.
- P4. For every network, H2 selects input feature 26.

| Network | Edit | Change on B (pp) | Change on C (pp) |
|---|---|---|---|
| Subject | 2A, as submitted | +0.39 [+0.33, +0.45]* | +0.39 [+0.35, +0.43] |
| Subject | 2A, edits 1–2 | +0.41 [+0.35, +0.47]* | +0.39 [+0.35, +0.44] |
| Subject | H1 | +0.40 [+0.34, +0.46]* | +0.39 [+0.34, +0.43] |
| Subject | H3 | — | +0.45 [+0.40, +0.49] |
| Subject | H2 | +0.09 [+0.07, +0.12]* | +0.07 [+0.05, +0.09] |
| Subject | H2+cal | +0.21 [+0.18, +0.24]* | +0.18 [+0.16, +0.20] |
| Seed 1 | H1 | +0.46 [+0.40, +0.53] | +0.41 [+0.37, +0.46] |
| Seed 1 | H3 | +0.51 [+0.45, +0.57] | +0.47 [+0.43, +0.51] |
| Seed 1 | H2 | +0.07 [+0.04, +0.09] | +0.06 [+0.04, +0.08] |
| Seed 1 | H2+cal | +0.21 [+0.17, +0.24] | +0.18 [+0.16, +0.21] |
| Seed 2 | H1 | +0.45 [+0.40, +0.51] | +0.46 [+0.42, +0.51] |
| Seed 2 | H3 | +0.52 [+0.46, +0.58] | +0.51 [+0.46, +0.55] |
| Seed 2 | H2 | +0.08 [+0.05, +0.10] | +0.07 [+0.05, +0.09] |
| Seed 2 | H2+cal | +0.17 [+0.14, +0.21] | +0.18 [+0.16, +0.20] |

\* Known before the pre-registration; repeated from the editing table. Unedited accuracy on B / C: subject 70.05 / 71.10%, seed 1 70.00 / 70.97%, seed 2 70.04 / 71.01%. Sources: `summary_out.txt`, `confirm_summary.json`, `final_table.txt`, `日志/confirm_all.log`. Intervals from a bootstrap over speakers, also in `summary_out.txt`, lead to the same conclusions.

All four predictions held. For P1, the difference between H1 and 2A's edits 1–2 on C is −0.01 pp [−0.03, +0.01]. P4 held but carries no information: H2 ranks input features using statistics of A only, so its choice cannot depend on the network.

### Where the gain of H1 comes from

H1 was evaluated on frames at different distances from the training data: frames the network had been trained on ("seen"), frames from the same training utterances that it had not yet been trained on ("unseen"), and the held-out sets B and C. Seen and unseen frames are samples of 1,000,000 frames each. Each cell gives the change with compensation / without compensation, in pp.

| Network | Seen training frames | Unseen training frames | B | C |
|---|---|---|---|---|
| Subject network | −0.11 / −0.01 | +0.16 / +0.25 | +0.40 / +0.44 | +0.39 / +0.46 |
| Subject network after 10,000 further steps at learning rate 1e-4 | −0.78 / −0.82 | −0.30 / −0.34 | +0.16 / +0.09 | +0.14 / +0.10 |

Source: `train_set_h1_out.txt`.

In both networks and in both versions of H1, the gain increases with distance from the training data: it is lowest on frames the network was trained on, higher on unseen frames of the training utterances, and highest on the held-out utterances. For the subject network with compensation, the difference between seen and unseen frames is −0.28 pp [−0.34, −0.21]. This shows where deleting the far-context weights helps and where it hurts. It does not by itself explain why training leaves these weights in place, and this repository makes no further claim about that.

## Limitations

- One subject network was shown to the language model, one model was used, and each condition was run in a single session. The results describe these sessions; they do not establish how the model behaves in general or how much its behaviour varies between sessions.
- The editors were told how the network was trained, including that it is undertrained; the readers were not. The editing results may depend on this information.
- The model snapshot and sampling settings of the sessions could not be fixed or recorded.
- The rules in step 6 were designed after the results on B were known. Only the confirmatory test on C and the new seeds is pre-registered. The supplementary analyses are post hoc, although their predictions were written down before they were run.
- C shares speakers with the training set; B does not.

## Repository layout

Folder and file names are in Chinese, as in the experiment itself: the prompts, task descriptions and file names that the language model saw were in Chinese. The folder names also stay in Chinese because the scripts use them as paths (for example `3_评估_勿给编辑者/结果/`), so renaming them would mean changing the code. Every Chinese text file has an English translation next to it; see [English translations](#english-translations) below.

| Folder | Meaning | Contents |
|---|---|---|
| `1_阶段1_读权重/` | Stage 1: reading the weights | Prompts `提示词_1A_只看结构.md` (1A, architecture only) and `提示词_1B与1C_读权重.md` (1B and 1C, reading weights); the answers `回答_1A.md`, `回答_1B.md`, `回答_1C.md`; the layout reconstructed by 1B, `1B_recovered_layout_28x81.csv`. |
| `2A_阶段2_改权重_零反馈/` | Stage 2A: editing, no feedback | Task description `任务说明.md`; the editor's tool file `tools.py`; the files the editor returned: edit report `修改说明.md` (diagnosis, edits, prediction) and code log `代码记录.md` (every piece of code it ran, in order). |
| `2B_阶段2_改权重_有反馈/` | Stage 2B: editing, with feedback | As for 2A, plus `eval_log.jsonl`, the record of its evaluations on A. |
| `3_评估_勿给编辑者/` | Evaluation (never shown to the editors) | Everything on the experimenter's side. Run all scripts from this folder. |
| `3_评估_勿给编辑者/代码/` | Code | All scripts (listed under Reproducing). |
| `3_评估_勿给编辑者/结果/` | Results | Evaluation reports `评估报告_*.md`; the stage-1 answer key and rubric `阶段1_答案.md`; the pre-registration `预注册_确认性检验.md`; the settings and predictions of the supplementary analyses `预测_*.md`; the raw outputs of the scripts (`*_out.txt`, `*.json`); per-utterance counts of correct frames (`*counts*.npz`, `per_utt_correct.npz`). |
| `3_评估_勿给编辑者/日志/` | Logs | Training logs, including the two abandoned configurations (`*_abandoned.txt`). |
| `3_评估_勿给编辑者/网络/` | Networks | Not stored in git; included in `weights.zip` (see below). |
| `3_评估_勿给编辑者/数据/` | Data | Not included; created by the packing scripts. |

The three files `预测_*.md` hold the settings and predictions of the supplementary analyses: `预测_第三版补充实验.md` for the generic reading algorithms and H1 on further-trained networks, `预测_第四版补充实验.md` for the 10,000-step training, and `预测_第五版补充分析.md` for the weight-displacement decomposition and H1 on training frames.

Files at the top level:

| File | Contents |
|---|---|
| [`WEIGHTS.md`](WEIGHTS.md) | The weight files in `weights.zip`: name, purpose, size, SHA-256 |
| [`WEIGHTS_SHA256SUMS.txt`](WEIGHTS_SHA256SUMS.txt) | The same checksums, in the format read by `shasum -c` |
| [`PREREGISTRATION_SHA256.txt`](PREREGISTRATION_SHA256.txt) | SHA-256 of the pre-registration and prediction files |

### English translations

The translations sit next to the originals and are named after them:

- `<name>.en.md` translates `<name>.md`: the prompts and task descriptions, the readers' answers, the editors' reports (`修改说明.md`) and code logs (`代码记录.md`), the evaluation reports, the stage-1 answer key, and the pre-registration and prediction files.
- `<name>.en.txt` translates a script output or log `<name>.txt` or `<name>.log` in `结果/` or `日志/`.
- `tools_usage.en.md` in each stage-2 folder explains `tools.py` in English.

The Chinese originals are the authoritative record. They are unchanged, and the translations were made after the experiment. The translations keep every number, file name, variable name and piece of code as in the original, and each translation was checked to contain the same set of numbers as its original. In the code logs, only the comments and docstrings inside code blocks are translated. `tools.py` is left byte-for-byte as the editors received it. In the scripts under `代码/`, the comments and docstrings are in English; the strings that the scripts print are still in Chinese, so that rerunning a script reproduces the saved output files exactly.

The JSON results contain Chinese only in a few `verdict` fields, which state the outcome under the decision rules fixed in advance:

| File | Field | Chinese | English |
|---|---|---|---|
| `displacement.json` | `verdict[0]` | 有一个弱的、指向 0 的拉力，但被随机运动主导 | There is a weak pull towards 0, but it is dominated by random motion |
| `displacement.json` | `verdict[1]` | 位移按 √T 增长，和随机游走一致 | The displacement grows as √T, consistent with a random walk |
| `noise_long_summary.json` | `verdict` | 优化在这段训练里没有去掉障碍（4 格区间全部 > 0） | Optimization did not remove the obstacle within this training (intervals of all 4 cells > 0) |
| `train_set_h1.json` | `verdict` | 支持过拟合解释 | Supports the overfitting explanation |

`2B_阶段2_改权重_有反馈/eval_log.jsonl` is unchanged; its two `note` fields are translated in that folder's `tools_usage.en.md`.

## Reproducing

### Requirements

Python 3 with NumPy and PyTorch. The reading baselines and the supplementary analyses also need SciPy and scikit-learn. The editors' `tools.py` needs only NumPy.

### Weights

The weights are not stored in git. Download `weights.zip` from the [Releases page](../../releases) of this repository, unzip it at the repository root, and check the files against the checksums stored in the repository:

```bash
unzip weights.zip
shasum -a 256 -c WEIGHTS_SHA256SUMS.txt
```

Every file lands at the path where the scripts expect it. [`WEIGHTS.md`](WEIGHTS.md) lists what each file is. Some files are byte-identical copies, because each stage folder holds exactly what that session received.

### Stage 1 needs only the weights

- To repeat a reading session, paste a prompt from `1_阶段1_读权重/` into a new session and attach `net_q7.npz` (1B, permuted) or `net_k3.npz` (1C, original); 1A gets no attachment. Score the answer with `结果/阶段1_答案.md`. The permutation used to make `net_q7.npz` is `结果/阶段1_打乱顺序_net_q7.npy`, which is included in `weights.zip`.
- The scripts that use only weights are `reading_baseline.py`, `reading_baseline_generic.py`, `displacement.py`, `label_swap_test.py` and `signal_check.py`.

### Everything else needs the original features

The features and labels are not included. `<data root>` below is a folder with two subfolders, `train-clean-100/` (training data) and `dev-clean/` (development data), each containing `mfcc/` and `transcript/`. From `3_评估_勿给编辑者/`:

```bash
python3 代码/pack_data.py <data root>              # development data, and the 3,000-utterance subset of the abandoned runs
python3 代码/pack_train_chunks.py <data root> 0     # the 12,000 training utterances, in three chunks
python3 代码/pack_train_chunks.py <data root> 1
python3 代码/pack_train_chunks.py <data root> 2
python3 代码/split_dev.py 数据/原始打包/dev_all.npz  # -> 数据/dev_A.npz, 数据/dev_B.npz
python3 代码/pack_heldout_C.py <data root>          # -> 数据/test_C.npz
```

The editors received split A as two files, `dev_A_part1.npz` (the first 675 utterances of `数据/dev_A.npz`) and `dev_A_part2.npz` (the remaining 676), with the arrays `x`, `y` and `lens`. To rebuild them:

```python
import numpy as np
A = np.load('数据/dev_A.npz'); n = 675; f = int(A['lens'][:n].sum())
np.savez('dev_A_part1.npz', x=A['x'][:f], y=A['y'][:f], lens=A['lens'][:n])
np.savez('dev_A_part2.npz', x=A['x'][f:], y=A['y'][f:], lens=A['lens'][n:])
```

### Scripts

Run every script from `3_评估_勿给编辑者/`, for example `python3 代码/evaluate.py 网络/edited_2A.npz`.

| Script | Purpose | Output (in `3_评估_勿给编辑者/`) |
|---|---|---|
| *Main experiment* | | |
| `pack_data.py`, `pack_train_chunks.py` | Pack the features | `数据/原始打包/` |
| `split_dev.py` | Split the development data into A and B by utterance | `数据/dev_A.npz`, `dev_B.npz` |
| `train_subject.py`, `train_subject_lib.py` | Train the subject network and the references with the same optimizer state | `网络/subject_net.npz`, `reference_plus*steps.npz` |
| `train_reference_lowlr.py` | Reference: new AdamW state, learning rate 1e-4 | `网络/reference_lr1e-4_plus2000steps.npz` |
| `evaluate.py` | Evaluate a network on B with bootstrap intervals | `结果/评估报告_*.md` |
| `targeted_controls.py`, `null_tests.py` | Edits aimed at other targets; null distributions | `结果/controls_out.txt`, `null_*.txt` |
| `audit_baselines.py`, `final_table.py` | Fixed rules on B; the main table | `结果/audit_out.txt`, `final_table.txt`, `per_utt_correct.npz`, `网络/baselines/` |
| `signal_check.py` | Inspect the structure of the subject network's weights | printed |
| `wlib.py` | Shared code: data loading, context windows, forward pass | — |
| *Confirmatory test and additional checks* | | |
| `pack_heldout_C.py` | Pack test set C | `数据/test_C.npz` |
| `train_seed.py` | Train the seed-1 and seed-2 subject networks | `网络/subject_seed1.npz`, `subject_seed2.npz` |
| `train_reference_fresh_lr1e-3.py`, `train_reference_samestate_lr1e-4.py` | References separating learning rate from optimizer reset | `网络/reference_fresh_*`, `reference_samestate_*` |
| `heur.py` | The frozen rules H1, H2, H3, H2+cal | — |
| `confirm_eval.py`, `summarize_confirm.py` | Confirmatory test on B and C, and its summary | `结果/confirm_counts.npz`, `confirm_summary.json`, `summary_out.txt` |
| `reading_baseline.py` | Reading algorithms R1 and R2 | `结果/reading_baseline_out.txt` |
| `label_swap_test.py` | How well the label-mapping test used by 1C separates similar labels | `结果/label_swap_out.txt` |
| `f26_and_speaker.py` | Further checks of input feature 26; bootstrap over speakers on B | `结果/f26_speaker_out.txt` |
| *Supplementary analyses* | | |
| `reading_baseline_generic.py` | Generic reading algorithms G1, G2, G3a, G3b | `结果/reading_baseline_generic_out.txt` |
| `noise_test.py` | H1 on further-trained networks (run `eval`, then `summary`) | `结果/noise_counts.npz`, `noise_summary.json`, `noise_test_out.txt` |
| `train_reference_lowlr_long.py` | 10,000 steps at learning rate 1e-4 with a checkpoint every 2,000 steps | `网络/reference_lr1e-4_long_step*.npz` |
| `noise_long.py` | H1 at each checkpoint (run `eval`, then `summary`) | `结果/noise_long_counts.npz`, `noise_long_summary.json`, `noise_long_out.txt` |
| `displacement.py` | Decompose how the far-context weights moved during training; weights only | `结果/displacement_out.txt`, `displacement.json` |
| `train_set_h1.py` | H1 on seen and unseen training frames (run `check`, `eval`, then `summary`) | `结果/train_set_h1_out.txt`, `train_set_h1.json`, `train_set_h1_counts.npz` |

The training and packing scripts take the data root or the three training chunks as arguments; the first lines of each script describe its usage.

### Pre-registration and predictions

The pre-registration and prediction files in `结果/` are byte-identical to the files written during the experiment. Their SHA-256 values are listed in [`PREREGISTRATION_SHA256.txt`](PREREGISTRATION_SHA256.txt) and can be checked with `shasum -a 256 -c PREREGISTRATION_SHA256.txt`. Each file states in its first lines when it was written. The git history of this repository begins after the experiment, so commit dates do not show when these files were written. The English translations of these files (`*.en.md`) were made after the experiment and are not covered by these checksums.

## License

- Code (`*.py`): MIT License, see [`LICENSE`](LICENSE).
- Weights (`weights.zip`) and results (all other files in this repository, including the results, logs, prompts, task descriptions and saved model outputs): [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- Data: not included.

## Citation

No paper is available yet. To cite this work, please refer to this repository by its URL and the release or commit you used. The question tested here comes from:

```bibtex
@misc{takehi2026,
  author = {Takehi, Rikiya and Teoh, Jayden and Zhang, Alex and Khattab, Omar},
  title = {Sparks of Weight-Space Programming: {LLMs} can natively read \& write neural network weights},
  howpublished = {Blog post, \url{https://rikiyat.github.io/blog/weight-space-programming/}},
  year = {2026},
  month = nov
}
```
