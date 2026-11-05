English translation of `评估报告_edited_2A.md`. The Chinese original is the authoritative record.

---

# Evaluation report: edited_2A.npz

Test set B: 1352 utterances, 980855 frames

| Network | Accuracy on B | Relative to the original network | 95% confidence interval (utterance bootstrap) |
|---|---|---|---|
| Original network (subject) | 0.7005 | — | — |
| **Edited** | **0.7044** | **+0.0039** | [+0.0033, +0.0045] |
| Reference: further training for 500 steps (lr 1e-3, 512,000 frames) | 0.7005 | -0.0000 | [-0.0012, +0.0010] |
| Reference: further training for 1000 steps (lr 1e-3, 1,024,000 frames) | 0.7024 | +0.0019 | [+0.0007, +0.0031] |
| Reference: further training for 2000 steps (lr 1e-3, 2,048,000 frames) | 0.7057 | +0.0052 | [+0.0040, +0.0064] |
| Reference: further training for 4000 steps (lr 1e-3, 4,096,000 frames) | 0.7068 | +0.0063 | [+0.0051, +0.0076] |
| Reference: lr lowered to 1e-4, then 2000 further steps | 0.7290 | +0.0285 | [+0.0276, +0.0295] |
| Control: random edits ×20 (same count, same norm) | mean 0.1123 | mean -0.5882, range [-0.6692, -0.4415] | — |

## Size of the edit

| Tensor | Entries changed | Fraction | Norm of the change | Norm of the original tensor |
|---|---|---|---|---|
| W1 | 516096 | 22.22% | 28.7103 | 109.0905 |
| b1 | 0 | 0.00% | 0.0000 | 26.9784 |
| W2 | 0 | 0.00% | 0.0000 | 67.6245 |
| b2 | 1024 | 100.00% | 38.4816 | 4.2614 |
| W3 | 12684 | 29.49% | 2.3495 | 8.3379 |
| b3 | 42 | 100.00% | 70.6978 | 1.9104 |
| Total | 529846 | 15.51% | | |
