# Weights (release asset `weights.zip`)

The weight files are **not stored in git**. Download `weights.zip` from this repository's Releases page and unzip it at the repository root; each file lands at the path below. Then verify:

```bash
shasum -a 256 -c WEIGHTS_SHA256SUMS.txt
```

Every `.npz` file holds the float32 arrays `W1` (2268×1024), `b1` (1024), `W2` (1024×1024), `b2` (1024), `W3` (1024×42) and `b3` (42) of the network

```
h1 = relu(x @ W1 + b1);  h2 = relu(h1 @ W2 + b2);  logits = h2 @ W3 + b3
```

The input `x` is 81 frames × 28 MFCC features, flattened as index `t*28 + f` (time outer, feature inner); frame `t = 40` is the frame being classified. Some files are byte-identical copies, because each folder holds exactly what that session received.

29 files, 383 MB in total.

| File | Purpose | Size (MB) | SHA-256 |
|---|---|---|---|
| `1_阶段1_读权重/net_k3.npz` | Stage 1, 1C: the subject network as given to the reader (identical to `2A_阶段2_改权重_零反馈/subject_net.npz`, `2B_阶段2_改权重_有反馈/subject_net.npz`, `3_评估_勿给编辑者/网络/subject_net.npz`) | 13.7 | `33ddf31589b6492e1814521c9b9a306e7311ac6d0b9aec61347ede28d38e86e0` |
| `1_阶段1_读权重/net_q7.npz` | Stage 1, 1B: the subject network with its 2268 inputs secretly permuted | 13.7 | `87ae64f9f1e30d7eda6b17ca108d6d4de3403f649b78823e255b056168a68b97` |
| `2A_阶段2_改权重_零反馈/edited_net.npz` | Stage 2A: the edited network returned by the editor (identical to `3_评估_勿给编辑者/网络/edited_2A.npz`) | 13.7 | `20dbc511ab772710148a54615df363a5a012cac1dcf0e9d74b41187ff00b80ed` |
| `2A_阶段2_改权重_零反馈/subject_net.npz` | Stage 2A: the network handed to the editor (identical to `1_阶段1_读权重/net_k3.npz`, `2B_阶段2_改权重_有反馈/subject_net.npz`, `3_评估_勿给编辑者/网络/subject_net.npz`) | 13.7 | `33ddf31589b6492e1814521c9b9a306e7311ac6d0b9aec61347ede28d38e86e0` |
| `2B_阶段2_改权重_有反馈/edited_net.npz` | Stage 2B: the edited network returned by the editor (identical to `3_评估_勿给编辑者/网络/edited_2B.npz`) | 13.7 | `b004430f7190e8c9bf0ab84e2b0f8bd7dc10ee7f93692008e2e0db708b0c77c0` |
| `2B_阶段2_改权重_有反馈/subject_net.npz` | Stage 2B: the network handed to the editor (identical to `1_阶段1_读权重/net_k3.npz`, `2A_阶段2_改权重_零反馈/subject_net.npz`, `3_评估_勿给编辑者/网络/subject_net.npz`) | 13.7 | `33ddf31589b6492e1814521c9b9a306e7311ac6d0b9aec61347ede28d38e86e0` |
| `3_评估_勿给编辑者/结果/阶段1_打乱顺序_net_q7.npy` | Stage 1 answer key: the input permutation used to make net_q7.npz (int64, length 2268) | 0.0 | `4be1e44601c7fb9dc8933569dfe66655c53642d1e8126a065dca28726414ef93` |
| `3_评估_勿给编辑者/网络/baselines/base_magprune22.npz` | Baseline: magnitude pruning of as many W1 weights as 2A changed + compensation | 13.7 | `54d054cbbbf058a871eac4389eab95366ac3322d3fb085a4fa6b80375d61ec48` |
| `3_评估_勿给编辑者/网络/baselines/base_rownorm18.npz` | Baseline: zero the 18 time offsets with the lowest mean row norm + compensation (the rule later frozen as H1) | 13.7 | `c461f2f7f48c3c7971ba134ab9d28523c677bc2d8c3c80b14a618a224a9d56db` |
| `3_评估_勿给编辑者/网络/baselines/combo_2A_E12_plus_f26.npz` | Post-hoc combination: 2A's edits 1+2 plus removing input feature 26 | 13.7 | `e00b529c41a16ad46a2000feb0172ede9d114d72ed2b9a226fc1b780ee9699f0` |
| `3_评估_勿给编辑者/网络/baselines/ed2A_E12.npz` | Editor 2A's edits 1+2 only (W3, b3 restored) | 13.7 | `c5676b5ed06deca7bc4f5c5f2ef2c4e8192e732d9d999adc63f2d7c67e5cbb82` |
| `3_评估_勿给编辑者/网络/edited_2A.npz` | Edit submitted by editor 2A (identical to `2A_阶段2_改权重_零反馈/edited_net.npz`) | 13.7 | `20dbc511ab772710148a54615df363a5a012cac1dcf0e9d74b41187ff00b80ed` |
| `3_评估_勿给编辑者/网络/edited_2B.npz` | Edit submitted by editor 2B (identical to `2B_阶段2_改权重_有反馈/edited_net.npz`) | 13.7 | `b004430f7190e8c9bf0ab84e2b0f8bd7dc10ee7f93692008e2e0db708b0c77c0` |
| `3_评估_勿给编辑者/网络/edited_2B_E1only.npz` | Editor 2B's edit 1 only (ablation) | 13.7 | `ee25e2c1c1ed16b279569213df52e4d1679dc64bfe97ebdf8dc05288423e7213` |
| `3_评估_勿给编辑者/网络/reference_fresh_lr1e-3_plus2000steps.npz` | Reference: 2,000 further steps, fresh AdamW, lr 1e-3 (2x2 control) | 13.7 | `cc287e127cfda60a73c6255a7ad07edb74a1a5fb4fb4aa38916ecd169404a3fc` |
| `3_评估_勿给编辑者/网络/reference_lr1e-4_long_step10000.npz` | Checkpoint at step 10000 of 10,000 further steps, fresh AdamW, lr 1e-4 | 13.7 | `3596a1b209a1ef16d92b99a230607a720ca509ef72e7693c4f6b77c73348743b` |
| `3_评估_勿给编辑者/网络/reference_lr1e-4_long_step2000.npz` | Checkpoint at step 2000 of 10,000 further steps, fresh AdamW, lr 1e-4 (identical to `3_评估_勿给编辑者/网络/reference_lr1e-4_plus2000steps.npz`) | 13.7 | `4eefe27f27f4c8f1e8d96b78bc6dd58352cee001dcedd8dc6abea2cc20095b96` |
| `3_评估_勿给编辑者/网络/reference_lr1e-4_long_step4000.npz` | Checkpoint at step 4000 of 10,000 further steps, fresh AdamW, lr 1e-4 | 13.7 | `029a02ddb557ed9e7a71af80ef07a04935f2760df20defd14b085f675fcb0e42` |
| `3_评估_勿给编辑者/网络/reference_lr1e-4_long_step6000.npz` | Checkpoint at step 6000 of 10,000 further steps, fresh AdamW, lr 1e-4 | 13.7 | `9c684d7478953da157ea13b97b7e686c25e6007c8f91c728b714777df94e642b` |
| `3_评估_勿给编辑者/网络/reference_lr1e-4_long_step8000.npz` | Checkpoint at step 8000 of 10,000 further steps, fresh AdamW, lr 1e-4 | 13.7 | `457965dc743380b126a059bafc06bfa6d42ba52321395e9a1e301e762daab3fc` |
| `3_评估_勿给编辑者/网络/reference_lr1e-4_plus2000steps.npz` | Reference: 2,000 further steps, fresh AdamW, lr 1e-4 (identical to `3_评估_勿给编辑者/网络/reference_lr1e-4_long_step2000.npz`) | 13.7 | `4eefe27f27f4c8f1e8d96b78bc6dd58352cee001dcedd8dc6abea2cc20095b96` |
| `3_评估_勿给编辑者/网络/reference_plus1000steps.npz` | Reference: 1000 further steps at constant lr 1e-3 (same run as the subject) | 13.7 | `8a550ad1e2a0c1d4a96f28fe0c040a082ad7e145244f2dfe463f64f3fbcd9305` |
| `3_评估_勿给编辑者/网络/reference_plus2000steps.npz` | Reference: 2000 further steps at constant lr 1e-3 (same run as the subject) | 13.7 | `61f5b9f2197e7df9a27611889633e477560577ccc00dac137d8e40d1719eb041` |
| `3_评估_勿给编辑者/网络/reference_plus4000steps.npz` | Reference: 4000 further steps at constant lr 1e-3 (same run as the subject) | 13.7 | `c205dfa63beb3075c4961ac166786ba49cd59d55c3d9524d27ffe7fe68474906` |
| `3_评估_勿给编辑者/网络/reference_plus500steps.npz` | Reference: 500 further steps at constant lr 1e-3 (same run as the subject) | 13.7 | `9fc8a9d9ae6879518adecaeba2e8c4eed432d2f105b1f8876571d8eccc284145` |
| `3_评估_勿给编辑者/网络/reference_samestate_lr1e-4_plus2000steps.npz` | Reference: 2,000 further steps, same AdamW state, lr 1e-4 (2x2 control) | 13.7 | `6dc739c75f84a30e30b8a1c945bf06749252316073daa8f587a3596467e4238a` |
| `3_评估_勿给编辑者/网络/subject_net.npz` | The subject network (AdamW, lr 1e-3, stopped at step 9,400 at about 70% accuracy) (identical to `1_阶段1_读权重/net_k3.npz`, `2A_阶段2_改权重_零反馈/subject_net.npz`, `2B_阶段2_改权重_有反馈/subject_net.npz`) | 13.7 | `33ddf31589b6492e1814521c9b9a306e7311ac6d0b9aec61347ede28d38e86e0` |
| `3_评估_勿给编辑者/网络/subject_seed1.npz` | Subject network retrained with seed 1 (confirmatory test) | 13.7 | `02944bbb56d9d16ada54e7dc5cd7729efc117ce3be7b333304cef61ad20f1fbd` |
| `3_评估_勿给编辑者/网络/subject_seed2.npz` | Subject network retrained with seed 2 (confirmatory test) | 13.7 | `af90b53b5427229d6836cd53a2aa5401a6dc89a0ede1a1666b0cb629982ec7ef` |
