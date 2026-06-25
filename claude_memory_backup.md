# Claude Code Memory Backup

## File: MEMORY.md

# Memory Index

## Project
- [project_kan_wnet.md](project_kan_wnet.md) — KAN-WNETR project: goal, architecture, datasets, training pipeline
- [project_dataset_cut_progress.md](project_dataset_cut_progress.md) — 34ch dataset cutting stopped at sub09 SNR2 l4 c0
- [project_fecgsyndb_dataset.md](project_fecgsyndb_dataset.md) — fecgsyndb: 10 subs, 34ch (32 abdominal + 2 maternal ref), ch32/33 are ref channels
- [project_pcdb_dataset.md](project_pcdb_dataset.md) — PCDB Challenge 2013: 75 real-world records, 4ch, 1000Hz, fqrs annotations
- [project_training_state_v3.md](project_training_state_v3.md) — KAN_WNETR never trained to completion; all top runs are 25.8M unetr baseline; RTX 3060 12GB; metric-batch bias; 8ch≡fullch
- [project_adfecgdb_finetune.md](project_adfecgdb_finetune.md) — W-NETR pretrain-synthetic→finetune-ADFECGDB; data present; --finetune-ckpt added; MUST use z-score norm for transfer
- [project_thesis_9chapter_restructure.md](project_thesis_9chapter_restructure.md) — latex_doan thesis restructured 7→9 chapters; skeleton+bullets done, builds clean (54p); user fills prose

## User
- [user_device_deployment.md](user_device_deployment.md) — Model for single-electrode device, multi-ch training is for generalization

## Feedback
- [project_polarity_alignment_deferred.md](project_polarity_alignment_deferred.md) — Ch22 polarity flip deferred, revisit later
- [feedback_kill_before_run.md](feedback_kill_before_run.md) — Always kill running Python processes before launching new training run
- [feedback_conda_env.md](feedback_conda_env.md) — Always use `conda run --no-capture-output -n KANWNET_fecg python` for all project scripts
- [feedback_training_flags.md](feedback_training_flags.md) — Always include --npy-set and --exp-tag in every training launch
- [feedback_validate_on_1sub_first.md](feedback_validate_on_1sub_first.md) — Validate recipe changes on 1sub (~5-6hr) before committing to the ~2.5-day 10sub run

## Reference
- [reference_mossformer2.md](reference_mossformer2.md) — MossFormer2 (teacher's rec): SOTA monaural separation, strong fECG fit, integration plan


---

## File: feedback_conda_env.md

---
name: Use conda env KANWNET_fecg for training
description: Always use conda env KANWNET_fecg (not python/python3 directly) to run training scripts
type: feedback
originSessionId: 695f47e5-84f1-4c20-a3cf-03a602b7f1db
---
Always use `conda run --no-capture-output -n KANWNET_fecg python ...` to run training scripts.

**Why:** `python` and `python3` on this machine resolve to wrong/stub executables. The project's environment is `KANWNET_fecg` in anaconda3.

**How to apply:** Any time running train scripts, eval scripts, or any project Python code — prefix with `conda run --no-capture-output -n KANWNET_fecg python`.


---

## File: feedback_kill_before_run.md

---
name: Kill previous process before new run
description: Always kill running Python training processes before launching a new training run
type: feedback
---

Always kill any running Python training processes before starting a new training run.

**Why:** User explicitly requested this — prevents resource conflicts and OOM on 32GB RAM system.

**How to apply:** Before every `python train_wnetr_networks_v3.py` invocation, check `tasklist | grep python` and kill existing training processes first.


---

## File: feedback_training_flags.md

---
name: Training run must include dataset flags
description: Every training launch must specify --npy-set and --exp-tag to select the dataset
type: feedback
originSessionId: 695f47e5-84f1-4c20-a3cf-03a602b7f1db
---
Always include `--npy-set` and `--exp-tag` when launching any training run.

**Why:** Without these flags the script defaults to a different dataset tag (e.g. `1sub_k5_c5_kh75_4ch`), not the intended one. The user had to correct this twice.

**How to apply:** Before running any `train_wnetr_networks_v3.py` command, always include both:
```
--npy-set exp --exp-tag 1sub_k5_c5_kh75_34ch
```
(or whatever dataset the user specifies). Verify against the reference run meta before launching.


---

## File: feedback_validate_on_1sub_first.md

---
name: feedback-validate-on-1sub-first
description: Validate training-recipe changes on 1sub (cheap) before committing to the multi-day 10sub run
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cb06672d-5f5f-4f59-b5e3-9aae0924317f
---

When testing a training-recipe hypothesis (loss change, completing an undertrained pretrain, schedule tweak, etc.), run it on the **1sub** dataset first, not 10sub.

**Why:** 1sub-8ch is ~62k samples (~11× smaller than 10sub's ~720k) → ~975 iters/epoch at BS64, and it actually converges (plateaus) so the hypothesis is answerable. On 2026-06-21 the user redirected a 10sub pretrain launch to 1sub for exactly this reason.

**REAL throughput (measured via checkpoint-file mtimes, run_20260622_084942, kan_wnetr_base 38.87M, BS64, 1sub-8ch):** ~**7.6 min/warm-epoch** (epoch1 cold incl. startup = ~14.9 min) → full 100ep ≈ **~13 hr**. DO NOT trust the script's displayed it/s / ETA — it under-reports by ~45% because the per-iter timer excludes dataloader stall time (Windows mmap + worker I/O). Always measure real epoch time from checkpoint LastWriteTime deltas, not the log's it/s. At this real rate 10sub@100ep ≈ **~6 days** (off the table without fewer epochs / smaller model / BS tuning).

**How to apply:** default any new pretrain/recipe experiment to `--exp-tag 1sub_k5_c5_kh75_8ch`. Only scale to `10sub_k5_c5_kh75_8ch` after the 1sub result confirms the change helps, and flag the ~2.5-day cost before launching 10sub. See [[project_training_state_v3]].


---

## File: project_adfecgdb_finetune.md

---
name: adfecgdb-real-data-fine-tuning-w-netr-protocol
description: W-NETR trains synthetic then fine-tunes on real ADFECGDB; data present; --finetune-ckpt added to v3; MUST use z-score norm
metadata: 
  node_type: memory
  type: project
  originSessionId: cb06672d-5f5f-4f59-b5e3-9aae0924317f
---

## W-NETR paper training protocol (Almadani 2023, the paper this project extends)
Two-phase: **pretrain on synthetic FECGSYNDB → fine-tune on real ADFECGDB**. "Capacity to train on a
simulated set and fine-tune on a real set improves extraction accuracy" (§II-C, IV-B). Paper also: trains
synthetic on channels 11/19/22/25, tests on sub10, uses **Z-score normalization** (§III-A), Huber loss,
Adam lr 1e-4, batch 90, ViT-B16 (L=12, K=360), same RTX 3060. PCDB = evaluation only (no signal GT).

## ADFECGDB (present locally at D:\Code\KAN_WNET\ADFECGDB)
- 5 EDF records r01,r04,r07,r08,r10. Channels: `Direct_1` (= ch0 = **direct fetal scalp = GROUND-TRUTH fECG**) + Abdomen_1..4 (= mixture). 1 kHz → 250 Hz (decimate/4).
- `generate_dataset_real.py` cuts 74×1000-sample windows + half-offset (+500) augmented copy per (record, abdominal branch). Segments now generated in ADFECGDB/fecg_ground & mixture (2960 each).
- `ADFECGDB/gen_paths_v3.py` (NEW, I wrote it) makes v3-compatible path arrays tag `adfecgdb_real` (train 2368 / val 592), absolute paths, mecg=mixture placeholder (no separate mECG GT). Per-branch val hold-out: r01/r04/r08/r10.

## v3 implementation (DONE)
- Added `--finetune-ckpt <path>`: loads ONLY model weights from a synthetic-pretrained ckpt, fresh optimizer, start_epoch=1, EMA synced, and SKIPS resume-meta override so CLI dataset/recipe wins. (vs `--resume-ckpt` which inherits the old run's meta.)

## CRITICAL: must use z-score (`--normalize-mode std`) for cross-dataset transfer
`var_global` divides by absolute variance → bakes in each dataset's amplitude scale (synthetic norm std≈2.8 @ global_var 0.13; ADFECGDB std≈10000 @ global_var 1e-8 → 3500× mismatch → breaks fine-tuning). Z-score → unit variance for BOTH (verified ADFECGDB mix_std=0.999). So pretrain AND fine-tune must use `--normalize-mode std`. The current var_global runs can't be the fine-tune base. Also use `--channel-weighted-loss false` for ADFECGDB (channel IDs don't map to synthetic 0-33 scheme).

## SSIM metric bug (FIXED 2026-06-18)
`compute_ssim_1d` returned `num/(den+eps)` with absolute eps=1e-8. On real ADFECGDB (amplitudes ~1e-5 mV), denormalized `den` ~1e-22 << eps → SSIM forced to 0.000000 for all epochs (while F1/PSNR were fine and rising → model was learning; only the metric + best.pth selection were broken). Fixed to `num/den` (den already >0 via dyn>=eps floor). Verified: tiny-signal SSIM 0.0→0.96; synthetic-scale unchanged (0.98). Only affects real/tiny-amplitude eval; synthetic runs unchanged.

## Phase 2 first real-data result (run_20260618_063820, KAN base→ADFECGDB)
Zero-shot transfer (ep1): F1 0.548, SSIM 0.125 — synthetic-pretrained model already detects ~55% of real fetal R-peaks before real-data training. Climbing with fine-tuning.

See [[project_training_state_v3]], [[project_pcdb_dataset]].


---

## File: project_dataset_cut_progress.md

---
name: Dataset 34ch cutting progress
description: generate_dataset.py 34ch cutting stopped at sub09 SNR2 level4 c0 — resume from there
type: project
---

Last run of `generate_dataset.py` (34 channels, 0-33) stopped at:
- Subject 09 (index 8), SNR index 2 (06dB), level 4, c 0
- sub01-05: fully cut (34ch)
- sub06-08: fully cut (34ch)  
- sub09: partially cut — stopped at snr06dB_l4_c0
- sub10: only 8 original channels (1,8,11,14,19,22,25,32)

**Why:** Disk space issue (D: drive was full at 466GB). User freed 50GB and needs to resume.

**How to apply:** Set `START_SUB = '09'` in generate_dataset.py. The skip logic will automatically skip already-cut segments and continue from where it left off.


---

## File: project_fecgsyndb_dataset.md

---
name: fecgsyndb dataset structure
description: PhysioNet fecgsyndb v1.0.0 — 10 subjects, 34 channels (32 abdominal + 2 maternal ref), 5 SNR levels, signal details
type: project
---

## Fetal ECG Synthetic Database (fecgsyndb) v1.0.0

**Source:** PhysioNet — physionet.org/content/fecgsyndb/1.0.0/

### Structure
- **10 simulated pregnancies** (sub01–sub10)
- **34 channels total:** channels 0–31 = abdominal fECG, channels 32–33 = maternal reference ECG
- **5 SNR levels:** 0, 3, 6, 9, 12 dB
- **5 repetitions (levels)** per condition: l1–l5
- **6 cases (c):** c0–c5 (baseline, noise, fetal movement, HR accel/decel, uterine contraction, ectopic beats)
- **Total:** 1,750 synthetic recordings, 145.8 hours, 1.1M fetal peaks

### Signal specs
- Sampling rate: 250 Hz
- Duration per recording: 5 minutes (75,000 samples)
- Resolution: 16-bit, WFDB format
- Amplitude: 1–2mV (maternal), fetal is much smaller

### Signal types per recording
- `fecg1` — fetal ECG component (34ch)
- `mecg` — maternal ECG component (34ch)  
- `noise2` — noise component (34ch)
- Mixture = fecg + mecg + noise2

### Channel interpretation
- **ch0–ch31:** 32 abdominal electrodes — these contain fECG signal mixed with mECG
- **ch32–ch33:** 2 maternal reference channels — primarily mECG, very little/no fECG content
- This explains why ch32/ch33 have SSIM ~0.5-0.6 in training — they are maternal reference channels, NOT abdominal channels. The model cannot extract fECG from them because there is minimal fECG signal present.

### Naming convention
- Raw: `sub{XX}/snr{YY}dB/sub{XX}_snr{YY}dB_l{Z}_c{W}_{signal_type}`
- Processed segments: `sub{XX}_snr{YY}dB_l{Z}_c{W}_{signal_type}_{kh}_{ch}`
  - kh = segment index (0–74, each 1000 samples = 4 seconds)
  - ch = channel index (0–33)

### Current dataset splits (1sub, 34ch)
- kh 0–52 → train (70%)
- kh 53–63 → val (15%)  
- kh 64–74 → test (15%)

**Why:** Understanding channel 32/33 as maternal reference explains their poor SSIM — they should be excluded from aggregate metrics or dropped from training.

**How to apply:** When evaluating, compute aggregate SSIM for ch0–ch31 only. Consider training with 32ch (0–31) to avoid wasting capacity on non-informative channels.


---

## File: project_kan_wnet.md

---
name: kan-wnetr-project-overview
description: "Architecture, datasets, training pipeline (v2), loss functions, normalization modes, and key files"
metadata: 
  node_type: memory
  type: project
  originSessionId: cb06672d-5f5f-4f59-b5e3-9aae0924317f
---

## Goal
Separate fetal ECG (fECG) from maternal ECG (mECG) in mixed abdominal recordings using deep learning — non-invasive fetal monitoring for single-electrode device.

**Why:** Clinically important; replaces invasive fetal scalp electrodes.

**How to apply:** Frame all suggestions in context of ECG signal separation, not generic ML.

---

## PAPER FACTS (verified 2026-06-22 from official repo github.com/Almadani92/W-NETR-for-FECG-extraction — THIS repo is a fork; paper's real preprocessing is in local util/dataset.py)
- Paper (Almadani, IEEE JBHI 2023, 27(7):3198-3209) headline **0.9988 is an F1-SCORE on ADFECGDB, NOT SSIM**. Our comparable best = ADFECGDB **F1 0.9622** / PCDB F1 0.912. Follow-up Attention R2W-Net (2025): F1 0.9917 ADFECGDB / 0.9803 PCDB. NEVER compare our synthetic-pretrain SSIM to the paper's F1 — category error.
- **Paper normalization** (util/dataset.py L70-76): bandpass 3-90Hz order3 @250Hz, then **(x - mean(mixture)) / var(mixture)** for ALL of mix/fecg/mecg using the MIXTURE's mean+var (shared stats, mean-centered, divide by VARIANCE). PCDB real (util/dataset_pcdb.py L64): same but **×20** after. Segment 992 @250Hz, single channel ([...,0]).
- **Our v4 `std` mode** = identical EXCEPT divides by std not var (one op off — the closest match). v4 `var` = ÷var but NOT mean-centered. v4 `paper` mode = BUGGY two-stage variant, does NOT match real paper. `var_global` (global scalar) is unrelated.
- TESTED 2026-06-22 (CORRECTS an earlier wrong note): the paper's PER-SEGMENT /var does NOT transfer cross-dataset by itself. /var amplitude ∝ 1/segment-variance, which differs ~3000x between synthetic (var~O(1)) and real ADFECGDB (var tiny → /var blows the signal up). Fine-tuning a /var(paper_exact) pretrain on ADFECGDB with paper_exact norm BROKE (SSIM 0.076, MSE ~3e6 — out-of-distribution input scale; same fragility as var_global). The paper sidesteps this with a per-dataset rescale (×20 for PCDB, dataset_pcdb.py). std is scale-stable (always ~unit variance) → it transfers automatically and is the right choice for real-data deployment. /var's edge is confined to SYNTHETIC pretrain SSIM.
- CLAUDE.md says "0.5-90Hz" bandpass — WRONG; real band is **3-90Hz** (order 3).

---

## Architecture

### Core Model: `models/unetr_kan_wnetr.py` — `KANWNETR`
- Dual-branch UNETR transformer
- Branch 1 → mECG; Branch 2 → fECG (with tanh-gated subtraction from mECG branch)
- Returns `(mecg_pred, fecg_pred)` — both `[B, 1, 992]`

### Transformer: `models/kanvit.py` — `KANViT`
- 12 layers, 12 heads, hidden_size=360, mlp_dim=700, patch_size=16
- Returns `(output, [hidden_states])` for UNETR skip connections

### KAN Layers
- `models/fasterkan.py` — ReflectionalSwitchFunction (RBF basis, 8 grids, range [-2,2])
- `models/kantransformerblock.py` — replaces MLP FFN with FasterKAN

---

## Training Script: `train_wnetr_networks_v2.py`

### Key Defaults (as of 2026-04-04)
- **Model:** kan_wnetr (110.5M params)
- **Loss:** composite = 0.5×Huber + 0.5×(1-SSIM), fecg_only=True (weight 4500.0)
- **Normalize:** "var" (per-sample variance). Supports: none, std, var, var_global, paper
- **Optimizer:** AdamW, lr=1e-4, weight_decay=0.01
- **Scheduler:** cosine (options: none, plateau, cosine, cosine_restart)
- **Warmup:** 10 epochs
- **AMP:** enabled, **EMA:** enabled (decay=0.999)
- **Batch size:** 64, **Epochs:** 100
- **Channel-weighted loss:** disabled by default. Weights at line 59-60

### Channel Loss Weights (line 59-60)
```python
CHANNEL_LOSS_WEIGHTS = {ch: 1.0 for ch in range(34)}
CHANNEL_LOSS_WEIGHTS.update({22: 1.5, 11: 1.5, 25: 1.5, 32: 0.1, 33: 0.1})
```
Enabled via `--channel-weighted-loss` CLI flag or `CHANNEL_WEIGHTED_LOSS = True`.

### Normalization Modes
- `none` — no normalization
- `std` — per-sample: `(x - mean) / std`
- `var` — per-sample: `x / var(x)`
- `var_global` — single global variance from all training samples, divide all by that value (preserves relative amplitude between channels)
- `paper` — paper baseline normalization

### Loss Function (`compute_total_loss`, line ~1139)
- **composite mode:** `0.5 * Huber(fecg_pred, fecg) + 0.5 * (1 - SSIM_1d(fecg_pred, fecg))`, scaled by FECG_ONLY_WEIGHT=4500
- **legacy mode:** just Huber loss
- Channel weighting: per-sample weight multiplied on Huber component (SSIM not weighted)
- Also computes MSE for logging

### Data Pipeline
1. `.npy` path arrays → WFDB signal loading → optional bandpass (3-90 Hz) → normalization → 992-sample segments
2. Cached to NPZ for speed (under `cache_fecg/{tag}/`)
3. Channel ID extracted from filename for per-channel metrics and weighted loss

### Metrics Logged (per epoch CSV)
- train_loss, val_loss, train_mse, val_mse
- Pan-Tompkins: prec/rec/f1/acc for both mECG and fECG
- Aggregate psnr_fecg, ssim_fecg
- Per-channel: psnr_ch{0-33}, ssim_ch{0-33}

---

## Key Files

| File | Purpose |
|------|---------|
| `train_wnetr_networks_v2.py` | **Main training script** — composite loss, var_global norm, channel weights, EMA, AMP |
| `train_wnetr_networks.py` | Older v1 training script (still used by some notebooks) |
| `models/unetr_kan_wnetr.py` | KANWNETR dual-branch model |
| `models/kanvit.py` | KAN-ViT encoder |
| `models/fasterkan.py` | FasterKAN layers |
| `networks.py` | Standard UNETR builder + weight init |
| `networks_kan.py` | KAN-UNETR builder |
| `init.py` | CLI argument parser |
| `fetal-ecg-synthetic-database-1.0.0/generate_dataset.py` | Segment raw PhysioNet data |
| `fetal-ecg-synthetic-database-1.0.0/Dataset_gen2.py` | Generate train/val/test .npy path arrays |
| `PCDB/generate_dataset_pcdb.py` | PCDB real-world data preprocessing |
| `plot_training_progress.ipynb` | Per-channel SSIM/PSNR/loss plots |
| `evaluate_pcdb_by_run_id.ipynb` | PCDB evaluation (has var_global bug — not yet fixed) |
| `compare_runs.py` | Compare all runs sorted by best SSIM |

---

## Model Variants

| CLI key | Class | Notes |
|---------|-------|-------|
| `kan_wnetr` | `KANWNETR` | Primary dual-branch (default) |
| `wnetr_networks` | `My_build_WNETR` | Standard W-NETR from networks.py |
| `kan_wnetr_networks` | `My_build_KANUNETR` | KAN version from networks_kan.py |
| `unetr` | `UNETR` | Single-output MONAI UNETR |
| `unetr_seq` | Sequential UNETR | Two-pass: mECG first, fECG from residual |
| `kan_unetr_seq` / `kan_wnetr_seq` | SeqKANUNETR | Sequential KAN version |


---

## File: project_pcdb_dataset.md

---
name: PCDB (PhysioNet Challenge 2013) dataset
description: Real-world fetal ECG dataset — 75 records set-a, 4ch abdominal, 1000Hz, 1min each, fqrs annotations
type: project
---

## PhysioNet/CinC Challenge 2013 — Noninvasive Fetal ECG

**Source:** physionet.org/content/challenge-2013/1.0.0/
**Purpose:** Detect fetal QRS complexes from noninvasive abdominal recordings, estimate fetal HR/RR/QT intervals.

### Dataset structure
- **Set A (training):** 75 records (a01–a25 original + a26–a75 supplementary) — signals + fqrs annotations
- **Set B (open test):** 100 records — signals only, no annotations
- **Set C (hidden test):** unpublished, for scoring

### Signal specs
- **4 channels:** AECG1–AECG4 (abdominal electrodes on mother)
- **Sampling rate:** 1000 Hz (needs resample to 250 Hz for KAN-WNETR)
- **Duration:** 1 minute per recording (60,000 samples per channel)
- **Format:** WFDB (.dat + .hea) + fqrs annotation (.fqrs)

### Annotations
- `.fqrs` — fetal QRS peak locations (binary WFDB annotation format)
- No maternal QRS annotations provided
- No ground truth fECG signal — only R-peak locations
- QT intervals available for subset with direct fetal ECG

### Local file structure
```
PCDB/
  set-a/       # a01–a25 (25 records)
  set-a-ext/   # a26–a75 (50 records)
```

### Key differences from fecgsyndb (synthetic)
| | fecgsyndb | PCDB |
|---|---|---|
| Type | Synthetic (simulated) | Real-world (clinical) |
| Channels | 34 (32 abdominal + 2 ref) | 4 (abdominal only) |
| Sampling rate | 250 Hz | 1000 Hz |
| Duration | 5 min/recording | 1 min/recording |
| Ground truth | Full signal (fecg, mecg, noise) | Only fetal R-peak locations |
| Subjects | 10 simulated | 75 real patients |
| Evaluation | PSNR/SSIM on signal | Precision/Recall/F1 on R-peaks |

### Usage in project
- Used as **real-world validation** — model trained on fecgsyndb, evaluated on PCDB
- Requires resampling 1000→250 Hz
- Evaluation: predict fECG signal → detect R-peaks → match with fqrs annotations
- Current best F1: ~0.508 (run_20260403_154021, 34ch) — low due to normalization bug in eval notebook

**How to apply:** When evaluating on PCDB, must handle domain gap (synthetic→real). Key issues: different noise characteristics, electrode placement, amplitude scale. Normalization must match training (var_global with correct global_var value).


---

## File: project_polarity_alignment_deferred.md

---
name: Ch22 polarity alignment deferred
description: User deferred Ch22 polarity flip preprocessing - may add later for multi-channel training
type: project
---

Ch22 fECG polarity alignment (flipping Ch22 fECG/mECG signs in build_cache) was planned but deferred by user.

**Why:** User wants to focus on other Phase 1 improvements first (global var norm, channel-weighted loss, per-channel metrics).

**How to apply:** When user asks to revisit multi-channel improvements or mentions Ch22 polarity, suggest implementing the flip in `train_wnetr_networks.py:build_cache()` after line 779.


---

## File: project_thesis_9chapter_restructure.md

---
name: project_thesis_9chapter_restructure
description: "LaTeX thesis (latex_doan) restructured 7→9 chapters per Dan-y outline; skeleton done, prose is bullets to fill"
metadata: 
  node_type: memory
  type: project
  originSessionId: f9dda8aa-52ea-4f50-b9c2-862e8a34f7d8
---

Thesis at `D:\Code\KAN_WNET\latex_doan` (main = `DoAn.tex`, chapters in `Chuong/*.tex`, subfiles class, biber/ieee). Restructured from 7→9 chapters on 2026-06-23 per outline `d:\Downloafd\Dan-y-KAN-WNETR (1).md`.

**9-chapter layout** (DoAn.tex \subfile refs):
1 `1_Gioi_thieu` · 2 `2_FECG_BG` · 3 `3_DL_Model` · 4 `4_CoSoLyThuyet` (NEW theory) · 5 `5_MoHinhDeXuat` (NEW, contribution only) · 6 `6_Dulieu_ThietLap` · 7 `7_Ket_qua` · 8 `8_HeThong` (NEW system) · 9 `9_Ket_luan`.

**Orphaned (content moved out, no longer \subfile'd):** `Chuong/4_Dataset.tex`, `Chuong/5_Trien_khai.tex` — safe to delete.

User chose: build skeleton first + write level = "khung + gạch đầu dòng" (I provide section structure + opening + bullet guidance; USER writes prose). Existing prose was preserved/moved, not rewritten. To-write spots marked `% TODO` + bullet lists; `[Cần điền số liệu thật]` for data.

**Ch8 source** = cloned repos `D:\Code\KAN_WNET\raspi-deploy` + `raspi-fecg-server` (ADS1293 soft-SPI driver, FecgProcessor ring-buffer 992/hop 200ms, ONNX INT8 ~29MB ~80-120ms/window 8-12× realtime, FastAPI+SQLite, FHR/MHR server-side Pan-Tompkins, 480×320 dashboard).

Build verified clean: `latexmk -pdf DoAn.tex` → 54 pages, 0 undefined refs/cites. Images were one dir too deep (`Chuong/Chuong/*.png`) — moved up to `Chuong/`.

Known TODO flagged in text: hyperparam table (80ep/bs64) contradicts prose (20ep/bs32) in Ch6 — must pick one. Ablation (Ch7.6) is priority-1 missing experiment. See [[project_training_state_v3]].


---

## File: project_training_state_v3.md

---
name: training-state-hardware-v3
description: KAN_WNETR never trained to completion; all top runs are the unetr baseline; RTX 3060 12GB; best configs
metadata: 
  node_type: memory
  type: project
  originSessionId: cb06672d-5f5f-4f59-b5e3-9aae0924317f
---

## Hardware
- GPU: **NVIDIA RTX 3060, 12 GB VRAM**. This is the binding constraint for batch size.
- `kan_wnetr` = 110.5M params (dual 12-layer ViT, hidden 360, mlp 700, num_grids 8).
- `unetr` baseline = 25.8M params. `kan_wnetr` is ~4.3× larger and ~4× slower.

## Critical insight (as of 2026-06-16)
**KAN_WNETR has never been trained to convergence.** Every top-of-leaderboard run is the
25.8M `unetr` baseline, NOT the research model `kan_wnetr`:
- Best 34ch run: `run_20260420_025212` — **unetr**, var_global, 100ep, BS32, channel-weighted → SSIM 0.9838 (ep94).
- Best ever: `run_20260225_225704` — **unetr**, single-channel ch19, 150ep → SSIM 0.979.
- Best `kan_wnetr` run: `run_20260403_154021` — abandoned at epoch 24, SSIM 0.963.
So KAN "looks worse" only because it was undertrained, not because it's inferior.

## Proven recipe (what made unetr hit 0.984 on 1sub_k5_c5_kh75_34ch)
- normalize_mode=var_global, loss_mode=composite (0.5 Huber + 0.5 SSIM), fecg_only=True
- AdamW lr=1e-4 (or 5e-5 @ BS192), cosine + warmup, AMP+EMA(0.999–0.9995)
- Dataset: single subject, 34 channels. Aggregate SSIM **includes ch32/33 (maternal ref)** which drag it down.

## Model variants & datasets (added 2026-06-17)
- **`kan_wnetr_base`** (new, in v3 build_model): hidden256/mlp512/grids4/12layers = **32.56M** params (vs 110.5M default `kan_wnetr`, 25.8M `unetr`). Device-deployable target. Trains BS128=7.2GB, BS64=3.8GB.
- **8ch dataset** `1sub_k5_c5_kh75_8ch` generated (Dataset_gen2.py --channels 1 8 11 14 19 22 25 32): train 62400 / val 13200 / test 14400. Channels = the project's "8 original channels"; ch32 is maternal-ref. ~4× smaller than 34ch.

## Results so far (8ch, 1sub_k5_c5_kh75_8ch, BS64, same recipe)
- **`kan_wnetr_base` × 8ch** (run_20260617_090136): plateaued ~ep80 at **abdominal SSIM 0.9215** (7ch excl ch32), aggregate 0.880, F1 0.927, PSNR 29.1. Stopped at ep91 (converged).
- No unetr 8ch run existed → comparison to unetr×34ch was cross-dataset/unfair (unetr had 4× data + full convergence: abd 0.994). So launched a real same-dataset unetr×8ch baseline.

## train_wnetr_networks_v4.py (created 2026-06-18) — KAN improvements
Forked from v3 for KAN experiments. Three research-backed KAN fixes (see deep-dive in convo):
1. FasterKAN SiLU **base/residual ENABLED** (was commented out) via opt-in `use_base_update` flag threaded through fasterkan→kantransformerblock→kanvit→unetr_kan_wnetr. v4 build_model passes use_base_update=True to kan_wnetr/base/small. v4 kan_wnetr_base = **38.87M** (vs v3 RBF-only 32.56M, +6.31M base_linear).
2. KAN models **need_init=False** (preserve FasterKAN Xavier spline init; global normal(0,0.02) was clobbering it).
3. **No weight decay on spline_linear** weights (AdamW param groups; 48 spline tensors).
**v3 is preserved/reproducible** (RBF-only): the shared model flag defaults False, so v3 builds KAN exactly as before. v3 got a top header (LAST USED 2026-06-18 + feature summary); v4 got a "what changed" header.
## RESULT: v4 KAN fixes WORK (2026-06-18) — closed ~85% of the KAN-vs-MLP gap
v4 kan_wnetr_base 8ch z-score pretrain (run_20260618_150853, plateau ep61) vs v3 RBF-only KAN vs unetr:
- **v4 KAN (hybrid): abd-SSIM 0.907**, agg 0.854, F1 0.925, PSNR 27.5
- v3 KAN (RBF-only): abd-SSIM 0.813, agg 0.720, F1 0.919, PSNR 26.7
- unetr (MLP): abd-SSIM 0.923, agg 0.868, F1 0.933
The 3 fixes (chiefly re-enabled SiLU base) lifted abd-SSIM 0.813→0.907 (gap to unetr 0.11→0.016). gnorm 2650→70 was the tell. KAN now ~matches MLP.
## 🏆 FINAL: v4 KAN BEATS the MLP baseline on REAL data (2026-06-18)
v4 KAN → ADFECGDB finetune 80ep (run_20260618_223311): **best F1 0.9462, SSIM 0.7569, PSNR 19.19** (still climbing at ep80).
Real-data ADFECGDB leaderboard (8ch z-score pretrain → 80ep finetune):
| model | synth abd-SSIM | real F1 | real SSIM | real PSNR |
| v4 KAN (hybrid) | 0.907 | **0.9462** | **0.7569** | 19.19 |
| unetr (MLP)     | 0.923 | 0.9426 | 0.7505 | **19.79** |
| v3 KAN (RBF-only)| 0.813 | 0.9383 | 0.7424 | 19.25 |
**The v4 KAN fixes made KAN_WNETR the top model on real fetal ECG F1 (0.946 > unetr 0.943).** Goal achieved: KAN now leads the MLP. (PSNR slightly behind unetr; F1+SSIM ahead.)

## Pushing further (2026-06-19)
- **more finetune epochs**: v4 KAN 8ch→ADFECGDB **150ep** (run_20260619_010233) → best **F1 0.9473** (ep91), **SSIM 0.7744** (ep150). F1 plateaus ~ep90 (80 was slightly short); SSIM keeps rising. Sweet spot ~90-100 ep.
- RUNNING: **34ch pretrain** v4 KAN z-score (run_20260619_015645, ~20-35 min/ep → ~15-25h to plateau; watcher armed). Tests "more data / all electrode positions".
- **34ch pretrain result: BROAD < FOCUSED (surprising).** v4 KAN 34ch pretrain (run_20260619_015645) plateaued ep40 at SSIM **0.828** on the shared 7 abd channels — WORSE than the 8ch pretrain's **0.907** on the same channels. The 34ch model spreads capacity across 34 channels (many noisy) → mediocre per-channel on the good ones. Not undertrained (2.8× more steps by ep40). Zero-shot transfer to ADFECGDB also worse (ep1 F1 0.472 vs 8ch 0.581). → **For specific/target electrode placements, focused 8ch pretrain beats broad 34ch.** "More data" hurt here via capacity dilution.
- DONE: 34ch→ADFECGDB finetune 150ep (run_20260619_174217) = **F1 0.9434, SSIM 0.7673** < 8ch's 0.9473. Confirmed: focused 8ch pretrain beats broad 34ch even after fine-tuning (34ch ~ties unetr 0.9426).
- VERDICT on the two levers: **more epochs HELPED (F1 0.9462→0.9473); 34ch more-data did NOT** (focused wins). Best KAN config so far: v4 fixes + 8ch z-score pretrain + ~90-150ep ADFECGDB finetune = **F1 0.9473 / SSIM 0.7744**, still > unetr (0.9426/0.7505).

### Gap-to-paper experiment (2026-06-19): 10sub × 8ch pretrain
Paper trains on 8 SUBJECTS (sub01-08); we'd only used 1 subject — the #1 real gap to paper's F1 0.9988. Generated `10sub_k5_c5_kh75_8ch` (Dataset_gen2 --num-subs 10 --channels 1 8 11 14 19 22 25 32): train=sub01-08 **720k samples** (0 missing), val=sub09, test=sub10 — paper's subject split + focused 8 channels (right "more data" = more SUBJECTS not more channels). v4 KAN 10sub pretrain run_20260619_193819 plateaued ep~16: synthetic agg-SSIM 0.670, shared-7 abd 0.732, F1 0.872 — LOWER than 1sub (abd 0.907). BUT that was healthy generalization, not capacity: **10sub→ADFECGDB zero-shot ep1 F1 = 0.679, BEST of all** (1sub 0.581, 34ch 0.472, unetr 0.437). **Subject diversity is the win — the single-subject data was the bottleneck, not the model.** ## 🏆 NEW PROJECT BEST (2026-06-20): 10sub→ADFECGDB = F1 0.9622
run_20260620_151427: **best F1 0.9622, SSIM 0.8192, PSNR 20.09** (ep150, still climbing). Beats ALL prior on all 3 metrics:
| model | F1 | SSIM | PSNR |
| 10sub×8ch v4 KAN | **0.9622** | **0.8192** | **20.09** |
| 1sub×8ch v4 KAN | 0.9473 | 0.7744 | 19.16 |
| 34ch v4 KAN | 0.9434 | 0.7673 | 19.33 |
| unetr MLP | 0.9426 | 0.7505 | 19.79 |
**Multi-subject pretrain was THE decisive lever** (+0.015 F1 over 1sub). Gap to paper 0.9988 narrowed ~5%→~3.7%. Best config: v4 KAN (FasterKAN hybrid + spline-init preserved + no-WD splines) + 8ch z-score pretrain on **10 subjects** + ~150ep ADFECGDB finetune. Remaining levers: more finetune epochs (still climbing), 110M KAN on 10sub (capacity+diversity).

## .qrs evaluation (2026-06-20) — gap to paper is REAL, not metric
Scored best 10sub model (run_20260620_151427) vs official ADFECGDB fetal-QRS annotations (±50ms): **F1 0.9644 all 20 rec·ch, 0.9612 held-out** — ≈ identical to PT-vs-PT 0.9622. So the metric was NOT the gap; the ~3.7% gap to paper's 0.9988 is real. BUT per-channel: good channels match paper (r01.4=0.9992, r08.4=0.9984, several ≈0.99); a few hard abdominal positions drag the mean (r04.1=0.865, r10.3=0.872). **Best-channel-per-record ≈ 0.986** — the device-relevant number (single well-placed electrode), ~1.3% from paper. Residual gap = training scale/capacity (→110M, more epochs), not metric/data-source. Eval method: load v4 ckpt, run full record in 992-windows (bandpass+zscore), pan_tompkins_fecg, match wfdb.rdann('rNN.edf','qrs')/4.

## Run-comparison caveats (researched 2026-06-17)
- **metric_max_batches bias:** 133 runs used mb=20, only 23 used mb=200. EVERY headline score (unetr 0.9839/0.9838, ch19 0.9793, old KAN 0.9633) is mb=20 = first ~640 val samples (shuffle=False → non-representative/easier subset), inflated vs full-val mb=200. Only compare same-mb runs. To fairly rank old best ckpts, re-eval at mb=200.
- **Dataset equivalence:** `8ch` ≡ `fullch` = identical 8 channels [1,8,11,14,19,22,25,32]; only `34ch` (all 34 ch, 265k samples) is the entire dataset. So 8ch experiments ≡ "1sub full" historical setup.

## KEY RESULT: unetr beats kan_wnetr_base on 8ch (same recipe/BS64)
Same-epoch, identical recipe: **unetr (26M) leads kan_wnetr_base (32M) at every epoch** (ep9: unetr abd 0.874 vs base 0.815). base plateaued at abd 0.9215; unetr@ep9=0.874 still climbing ~0.02/ep → will pass base's plateau. So KAN (as configured) underperforms the MLP baseline at comparable params.
**Suspected cause = grad-clip asymmetry:** KAN gnorm ≈2650 vs unetr ≈2.0; with --grad-clip 1.0, unetr ~unclipped but KAN renormalized ~1000×/step → throttled. **Next experiment: kan_wnetr_base × 8ch with --grad-clip 3000** to test if un-throttling closes the gap. If yes, apply to 110M.

## Active training queue — 4 runs, sequential, single GPU (kill-before-run, no concurrency)
W-NETR synthetic→real pipeline for BOTH models, z-score throughout (see [[project_adfecgdb_finetune]]).
Launch each when the prior completes; fill finetune `--finetune-ckpt` from that pretrain's best.pth.
Common pretrain flags: `--npy-set exp --exp-tag 1sub_k5_c5_kh75_8ch --normalize-mode std --batch-size 64 --epochs 100 --lr 1e-4 --scheduler-mode cosine --warmup-epochs 10 --use-amp true --use-ema true --ema-decay 0.9995 --fecg-only true --fecg-only-weight 1.0 --channel-weighted-loss true --composite-w-huber 0.5 --composite-w-ssim 0.5 --composite-w-deriv 0.2 --grad-clip 3000 --num-workers 8`
Common finetune flags: `--npy-set exp --exp-tag adfecgdb_real --normalize-mode std --batch-size 64 --epochs 40 --lr 2e-5 --warmup-epochs 2 --ema-decay 0.999 --fecg-only true --fecg-only-weight 1.0 --channel-weighted-loss false --composite-w-deriv 0.2 --grad-clip 3000 --num-workers 8 --finetune-ckpt <PRETRAIN_best.pth>`

1. ✅ DONE: KAN pretrain = **run_20260617_213716** (stopped ep70, plateaued; best.pth used for #2).
2. ✅ DONE: KAN→ADFECGDB finetune 40ep = **run_20260618_063820**: **F1 0.928, SSIM 0.661, PSNR 18.6 @ ep40** (zero-shot ep1 was F1 0.548/SSIM 0.125 → fine-tuning big lift). NOTE: still climbing at ep40 — **40 epochs undershot**, use ~80 for plateau.
3. ✅ DONE: unetr × 8ch std pretrain = **run_20260618_065411** (plateaued ep40, abd-SSIM **0.923**, f1 0.933). **unetr DECISIVELY beats kan_wnetr_base**: converged ~2× faster AND ~0.11 higher abd-SSIM (KAN plateau 0.813). Clean confirmation KAN-base underperforms MLP here.
4. ✅ DONE: unetr→ADFECGDB 80ep = run_20260618_102840: best F1 0.9426, SSIM 0.7505, PSNR 19.79.
5. ✅ DONE: KAN→ADFECGDB 80ep = run_20260618_105013: best F1 0.9383, SSIM 0.7424, PSNR 19.25.

## FINAL real-data verdict (matched 8ch z-score pretrain → ADFECGDB 80ep)
**unetr modestly > KAN base on real fetal ECG**: F1 0.943 vs 0.938, SSIM 0.751 vs 0.742, PSNR 19.8 vs 19.3. BUT gap is small (vs the big synthetic gap), because KAN transfers better zero-shot (ep1 F1 0.548 vs 0.437) and nearly catches up after fine-tuning. Both still creeping at ep80. Both ≈0.94 F1 — below paper's W-NETR 99.88% (paper used full 34ch, bigger model, sub-splits). KAN substitution is competitive but NOT beating MLP → motivates trying MossFormer2 [[reference_mossformer2]]. Headroom for higher: 34ch pretrain, full 110M KAN, more finetune epochs.
GOAL: compare KAN vs unetr on REAL ADFECGDB val. Paper's W-NETR hit F1 99.88% on ADFECGDB (full 34ch); our base/8ch/40ep prelim KAN = 92.8%.
## MossFormer2 integrated (3rd model) — 2026-06-18
- Added `mossformer2` to v3 build_model (vendored official ClearerVoice-Studio MossFormer2 separator into `models/mossformer2/`; deps installed: rotary_embedding_torch, torchinfo). Config dim256/16 blocks/k16/2spks = **14.63M params** (smaller than KAN 32M, unetr 25.8M). Wrapper: [B,1,T]→squeeze→MossFormer→(src0=mECG, src1=fECG). need_init=False.
- Added **SI-SNR loss** to v3 composite framework: `--composite-w-sisnr` + `si_snr_loss()` (fp32 inside for AMP stability, no PIT). Verified BS64 fwd+bwd 7.19GB.
- 6. RUNNING: MossFormer2 × 8ch z-score **pure SI-SNR** pretrain = run_20260618_121815 (lr1e-4, BS64, grad-clip 10000=un-throttled, fecg_only_weight1.0). ~1 it/s = **~16 min/epoch** (~4× slower than KAN/unetr; ~13-23h to plateau). AMP nan at start = GradScaler warmup, settled by ~ep1.
- 7. QUEUED: MossFormer2 → ADFECGDB finetune (80ep, SI-SNR, std, `--finetune-ckpt <run#6 best.pth>`, channel-weighted-loss false).
GOAL: 3-way real-data compare KAN(0.938) vs unetr(0.943) vs MossFormer2 on ADFECGDB F1/SSIM. See [[reference_mossformer2]].
NOTE: MossFormer2 uses its native SI-SNR (vs composite for KAN/unetr) — best-shot test, not loss-controlled. Composite-loss variant available if strict architecture-only comparison wanted.

## Earlier 8ch results (var_global, NOT transferable to real — superseded by z-score track)
- kan_wnetr_base×8ch var_global plateaued abd-SSIM 0.9215. unetr×8ch (stopped ep9) led same-epoch. clip3000 experiment (var_global) stopped early to pivot — grad-clip-helps-KAN hypothesis still untested.

## Why BS matters (lesson)
BS halves/doubles gradient steps/epoch. 110M@BS64 on 34ch converged slower per-epoch than the old KAN@BS32 (which hit 0.963@ep23). Prefer smaller batch (more steps) for the big model. unetr "0.9838" baseline = a 4-run resume chain (181710→004909→010237→025212), ~100ep from scratch, BS32.

## Empirical findings (run_20260616_193047, kan_wnetr, BS64, P1 loss fixes)
- **VRAM:** kan_wnetr (110.5M) at BS64 peaks 8.2 GB (BS80=10.85 GB tight). BS64 = safe sweet spot, no grad-accum needed.
- **Grad-clip is 100% saturated:** with FECG_ONLY_WEIGHT=4500, raw grad-norm ≈ **1.6e7** vs `--grad-clip 1.0` → clip renormalizes every step (does nothing useful; AdamW cancels the scale anyway). Clean fix for next run: `--fecg-only-weight 1.0 --grad-clip 1.0` so loss is O(1) and clip becomes a real spike-guard.
- **Epoch cost:** 1sub_34ch = ~265k samples/epoch → 4143 iters @ ~1.1 it/s ≈ **1 hr/epoch**; 100 ep ≈ 4 days. KAN ~4× slower than unetr baseline. Consider fewer epochs or right-sizing for iteration speed.

## v3 P1 loss changes (done 2026-06-16)
- Added `temporal_derivative_loss` term (COMPOSITE_W_DERIV, default 0.2) + optional Pearson `correlation_loss` (COMPOSITE_W_CORR, default 0).
- ch32/33 loss weight 2.0 → 0.1 (maternal-reference, ~no fECG).
- `--grad-clip`, `--composite-w-{huber,ssim,deriv,corr}` now CLI args; logged to run_meta + restored on resume; gnorm printed per-iter.

**Why:** Future "improve KAN_WNETR" work must first train it to completion before comparing to baseline. See [[project_kan_wnet]] and [[project_fecgsyndb_dataset]].

## v4 loss additions + try-1 experiment (2026-06-21)
- **train_wnetr_networks_v4.py gained 2 opt-in loss features** (default OFF → v3-identical):
  (1) `multi_resolution_stft_loss` (`--composite-w-stft`, FFT {128,256,512}, spectral-conv + log-mag L1, forced fp32 under AMP);
  (2) `base_recon_loss` so `--base-loss {l1,huber,mse}` now applies in COMPOSITE mode (was legacy-only).
- **Best ADFECGDB fine-tune baseline = `run_20260620_151427`**: kan_wnetr_base, finetune from `run_20260619_193819/best.pth` (synthetic 10sub 8ch `10sub_k5_c5_kh75_8ch`), lr 2e-5 cosine(w2,t0=20,×2), norm **std**, fecg_only_w 1.0, grad_clip 3000, composite huber0.5/ssim0.5/deriv0.2, BS64 → **F1 0.9622 / SSIM 0.8192 / PSNR 20.09** (plateaued flat last ~10ep).
- **TRY-1 `run_20260621_091701`** (L1 base + STFT@0.2 + BS32, else baseline) → **F1 0.9383 / SSIM 0.7317 / PSNR 18.82. LOST on all 3.**
  - **Root cause:** STFT@0.2 was 5× over-weighted — raw STFT≈2.1 so its contribution (~0.42) was the LARGEST composite term, drowning SSIM/morphology. L1 base compounded (de-emphasizes QRS peaks on sparse z-scored ECG) + caused AMP gnorm inf/nan overflow (L1 constant gradient).
- **LESSON:** weight a new composite term by its RAW magnitude, not nominal value. STFT needs ~0.05 to match deriv's contribution. Keep base_loss=huber (L1 hurt + destabilized AMP).
- Ablation `run_20260621_155306` (baseline + STFT@0.05 only, huber, BS64) → F1 0.9476 / SSIM 0.7561. Still < baseline 0.9622/0.8192, meta-diff confirms ONLY stft differs.
- **CRITICAL METHODOLOGY FINDING:** v4 (and v3) set NO random seed (`torch.manual_seed`/`np.random.seed` absent) + `cudnn.benchmark=True` → training is **fully nondeterministic**. On the tiny ADFECGDB fine-tune set (2368 samples, 150ep) run-to-run variance is large and UNMEASURED. So single-run fine-tune comparisons (incl. the 0.9622 "best") are confounded — can't attribute the STFT/L1 drops to the change vs noise. To do valid A/Bs: add a --seed flag + cudnn-determinism, or re-run baseline 2-3x to quantify spread. **User decided 2026-06-21 to bank 0.9622 and stop chasing fine-tune loss tweaks.**

## PCDB real-world generalization (2026-06-21) — KEY RESULT
- Eval tooling: `evaluate_pcdb_by_run_id.ipynb` (imports v3!) → ported to v4-compatible script (scratchpad `eval_pcdb_run.py`). PCDB set-a fully present: `PCDB/set-a` (25 rec) + `PCDB/set-a-ext` (50 rec) = 75 records; eval uses TABLE_SELECTED_LEADS (curated good leads) → 78 record/lead pairs, 1170 windows, R-peak F1 vs `.fqrs`, tol 50ms, std-norm + 3-90Hz bandpass.
- **MUST eval v4 runs through v4** (`build_model` sets use_base_update=True → 38.87M); v3's build_model gives 32.56M and silently drops base_linear (strict=False) → garbage.
- **Best model `run_20260620_151427` (std-norm, ADFECGDB fine-tune): PCDB F1 = 0.9121** (P 0.9298 / R 0.8950), macro-F1 0.9161. vs OLD var_global run `20260417_004909` = **0.4498**. → std-norm + real-data fine-tune is what enables cross-dataset (ADFECGDB→PCDB) generalization; var_global does NOT transfer. Saved to `logs/run_20260620_151427/pcdb_eval/`.
- Caveat: on curated good-lead set (same selection as prior evals, apples-to-apples). Worst pairs (a01-l4 0.617, a41-l1 0.672) drag aggregate; best hit 0.996.

## 1sub-8ch pretrain-to-convergence + paper_exact norm test (2026-06-22)
- **STD-norm baseline `run_20260622_084942`** (kan_wnetr_base, 1sub-8ch, from scratch, std norm, full recipe): plateaued **ep59 SSIM 0.8499 / F1 0.9248 / PSNR 27.34** (slope 0.00035/ep). NOTE: this synthetic-val SSIM includes ch32 (maternal ref, SSIM ~0.39) which drags it; fECG-only-channel SSIM ~0.885. real epoch time ~7.6min warm. best.pth saved for fine-tuning.
- KEY: the paper's headline 0.9988 is **F1 not SSIM** — see [[project_kan_wnet]] PAPER FACTS. Paper norm = (x-mean(mix))/var(mix); ours was /std. Added faithful `paper_exact` mode to v4 (NORMALIZE_MODE choices + cache-norm branch L1119+ + norm_tag zpaperx).
- **paper_exact retry** launching now (same recipe, only --normalize-mode paper_exact) to test if matching the paper's /var normalization changes SSIM/F1. Compare its plateau vs std baseline (0.8499/0.9248).
- **CEILING-LIFT TEST RESULT (2026-06-22): completing the pretrain on 1sub does NOT beat 10sub@ep16.** Fine-tune from the completed std 1sub pretrain (run_20260622_171304, ADFECGDB, BS32, lr 2e-5) plateaued **F1 ~0.949 @ep46** — BELOW the 0.9622 best (which came from the undertrained 10sub@ep16 pretrain). CONCLUSION: data DIVERSITY + total steps of 10sub (10 subjects, ~180k steps) beats convergence-on-1-subject (~57k steps). So the real ceiling-lift requires **completing the 10sub pretrain** (the ~2.5-day, now ~6-day @7.6min/ep run) — completion alone on limited data isn't enough. (Minor confound: fine-tune used BS32 vs baseline BS64.) Killed at ep67 (plateaued) to free GPU for the paper_exact run. **PCDB eval of this fine-tune (epoch_067.pth) = F1 0.829** vs best model's 0.912 — gap BLOWS UP cross-dataset (−0.083 PCDB vs −0.013 ADFECGDB), confirming 1sub pretrain overfits to one subject's morphology and generalizes poorly. So pretrain DATA DIVERSITY (10 subjects) >> convergence, especially for generalization. Real ceiling-lift = complete the 10sub pretrain.
- LESSON: force-killing a run mid-checkpoint-write CORRUPTS best.pth (truncated, "failed finding central directory"). The last epoch_NNN.pth (written just before best.pth) is the intact fallback. To eval/finetune a killed run, use epoch_NNN.pth, not best.pth.
- paper_exact (run_20260622_163247) EARLY signal: at ep6 SSIM 0.682 / F1 0.807 vs std's 0.597 / 0.799 — /var converging to HIGHER SSIM at matching epochs (needs plateau to confirm).
- **CONFIRMED (2026-06-22): paper's /var normalization (paper_exact) BEATS /std.** paper_exact run_20260622_163247 reached SSIM 0.8675 / F1 0.9289 / PSNR 27.76 by ep41 (still slowly creeping ~+0.0006/ep, asymptote ~0.875/0.93) vs std baseline run_20260622_084942 final 0.8499 / 0.9248 @ep59. So /var gives **+~0.018 SSIM, +~0.004 F1, and converges ~1.4-2x faster** (matched std's FINAL SSIM by ep27). KEY ACTIONABLE: paper_exact (per-segment (x-mean(mix))/var(mix)) should be the default pretrain norm, not std. NEXT: re-fine-tune ADFECGDB from paper_exact best.pth (REQUIRES --normalize-mode paper_exact at fine-tune AND a paper_exact branch in eval_pcdb_run.py preprocess) to test if it lifts real F1 above 0.9622 / PCDB 0.912.
- **RESULT (2026-06-22): the paper_exact fine-tune on ADFECGDB BROKE (run_20260622_225700).** /var amplitude on ADFECGDB (tiny variance) is ~3000x the synthetic /var amplitude the model trained on → out-of-distribution inputs → SSIM 0.076, MSE ~3e6, F1 garbage. Killed. CONCLUSION: /var(paper_exact) WINS synthetic pretrain SSIM (0.868 vs std 0.850) but does NOT transfer to real data without per-dataset amplitude matching (the paper's ×20-style hack). **std remains the right norm for real deployment; best model stays run_20260620_151427 = ADFECGDB F1 0.9622 / PCDB 0.9121.** To leverage the better /var pretrain for real F1 you'd need either (a) fine-tune it with std norm (amplitudes match: synth /var ~O(1) ≈ ADFECGDB std ~O(1)), or (b) add per-dataset amplitude matching for /var. The paper_exact pretrain best.pth (ep42, SSIM 0.8682) is saved at logs/run_20260622_163247.
- **FINAL /var verdict (2026-06-23, run_20260623_093308 = 1sub /var pretrain fine-tuned with STD norm, which transfers fine):** plateaued ADFECGDB F1 0.9507 (peak ep90) / PCDB F1 0.8372. vs 1sub STD pretrain ft (0.949 / 0.829) = within noise (+0.002/+0.008). So /var's synthetic-SSIM edge (0.868 vs 0.850) does NOT translate to a better real model. CONCLUSION across all experiments: /var = synthetic-metric win only; **pretrain DATA DIVERSITY (10 subjects) is THE lever** — both 1sub variants (~0.95 ADFECGDB / ~0.83 PCDB) sit far below the 10sub baseline (0.9622 / 0.9121). Normalization (std vs /var) and 1sub-convergence are NOT the ceiling-lift. **Best model UNCHANGED: run_20260620_151427 (10sub std→ft) = ADFECGDB 0.9622 / PCDB 0.9121.** The genuine next lever is completing the 10sub std pretrain (the ~6-day @7.6min/ep run) — that combines diversity + convergence.
- **HIGH-mb20 MIRAGE confirmed (2026-06-23, run_20260623_105156):** fine-tuned run_20260417_004909 (unetr/var_global/1sub-34ch, "SSIM 0.980" but that was mb=20-inflated; it scored PCDB 0.45 zero-shot) via std norm → ADFECGDB F1 0.9447 (ep87) / PCDB F1 0.8370. BELOW the 0.9622/0.9121 baseline, same cluster as all 1sub runs. Its var_global global_var was never saved → not even cleanly re-evaluable. CONCLUSION (4th confirmation): mb=20 numbers are inflated & NON-predictive of fine-tune quality; var_global high SSIM is a mirage; pretrain DATA DIVERSITY (10 subjects) is the only lever that moves real F1/PCDB. Don't fine-tune from high-mb20/var_global/1sub runs. Note: unetr ckpt loads cleanly in v4 build_model('unetr') (25.8M, 0 missing) and unetr trains fine in v4 (returns mecg/fecg tuple).


---

## File: reference_mossformer2.md

---
name: mossformer2-for-fetal-ecg-teacher-s-recommendation
description: SOTA monaural speech-separation model; strong fit for single-lead fECG separation; how to integrate
metadata: 
  node_type: memory
  type: reference
  originSessionId: cb06672d-5f5f-4f59-b5e3-9aae0924317f
---

## What it is
MossFormer2 (Zhao et al., Alibaba; ICASSP 2024, arXiv 2312.11825). Single-channel, **time-domain source separation** (Conv-TasNet/SepFormer family): learnable 1-D conv encoder → masking separator → 1-D conv decoder. Hybrid separator = **MossFormer gated single-head attention** (conv-augmented, joint local+global) + **RNN-free FSMN recurrent module** (gated conv units + dilated FSMN + dense connections). SOTA on WSJ0-2/3mix, Libri2Mix, WHAM!; +1.3 dB SI-SNRi over MossFormer. Trained with SI-SDR/SI-SNR loss + PIT. ~10–50M params.
- Code + pretrained: **ClearerVoice-Studio** (github.com/modelscope/ClearerVoice-Studio), separation model at `train/speech_separation/models/mossformer2/`, configs `config/train/MossFormer2_SS_{8K,16K}.yaml`.
- Audio samples repo: github.com/alibabasglab/MossFormer2.

## Why it fits fECG (teacher was right)
Single-lead fECG extraction = monaural 2-source separation (maECG → {mECG, fECG}), exactly speaker separation. Monaural + time-domain + masking + joint local-global attention + FSMN (cardiac periodicity) match all constraints incl. single-electrode device ([B,1,L]). No prior MossFormer-for-fECG found → novel transfer. Sources are distinguishable → **fixed output assignment, no PIT needed** (simpler than speech).

## Integration plan (v3 is model-agnostic: model(x)->(mecg,fecg))
1. Lift separator from ClearerVoice mossformer2 dir + 8K config.
2. Add `mossformer2` to build_model; wrap to 2 outputs → (out0=mECG, out1=fECG).
3. Retune conv encoder kernel/stride for 250 Hz ECG (defaults tuned for kHz speech), input 992.
4. Add SI-SNR loss option to v3 (separation-native, scale-invariant — pairs with z-score).
5. Run same pipeline: synthetic z-score pretrain → ADFECGDB fine-tune → 3-way compare vs [[project_training_state_v3]] KAN/unetr.

Caveats: ECG≠speech (250Hz, short 992 seq → fast/low-mem, fits 12GB easily); SI-SNR sensitive to tiny amplitudes (z-score helps, cf [[project_adfecgdb_finetune]] SSIM bug).


---

## File: user_device_deployment.md

---
name: Device deployment context
description: Model will run on a single-electrode device for fetal ECG extraction at various body positions
type: user
---

The KAN-WNETR model is intended for deployment on a physical device with a single electrode. The device takes signals from 1 channel at a time to analyze fetal ECG. Multi-channel training is for generalization across electrode placements, NOT for multi-channel fusion. Model input must remain [B,1,992].


---

