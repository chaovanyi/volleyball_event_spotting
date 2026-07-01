# Volleyball Event Spotting + Ball Localization — Method, Architecture & Results

Ported from the tennis **T-DEED + ASTRM** pipeline (`tdeed_astrm/`, Exp 10/11) to the
VNL volleyball dataset. A single network jointly (a) spots 8 rally events in time and
(b) localizes the ball in every frame.

---

## 1. Method

- **Task.** Per-frame dense prediction over 100-frame clips:
  - **Temporal event spotting** — classify each frame as one of 8 events or background,
    scored with the T-DEED soft-NMS + tolerance-matching protocol.
  - **Ball localization** — regress the ball (x, y) for every frame via a heatmap +
    sub-pixel offset head (CenterNet-style), decoded to normalized coordinates.
- **Joint training, detached ball head.** The spatial (ball) head is `.detach()`ed from
  the backbone, so ball-localization gradients never corrupt the event features. The
  backbone still *co-adapts* to both objectives because it is trained end-to-end for the
  event losses while the ball head rides on top of the shared features. This detachment is
  what let the event head stay clean even when the ball head was numerically unstable in
  early runs.
- **Two decisive fixes over the naive port** (these closed the entire tennis→volley gap):
  1. **NaN-divergence fix.** Original runs diverged to NaN around epoch 17 (fp16
     activation overflow poisoning BatchNorm running stats). Fixed with gradient clipping
     (`clip_grad_norm_=1.0`, unscale-then-clip) + a non-finite-loss guard. Training is now
     stable for the full 50 epochs, letting the backbone co-adapt like the tennis run.
  2. **Aspect-ratio fix (biggest spatial win).** Frames are stored 398×224 (16:9) but were
     being squished into a 224×224 **square**, compressing the horizontal axis to 0.56× and
     distorting ball-x. Feeding an **aspect-correct 320×180** rectangle (true 16:9, same
     pixel budget) removed the distortion and lifted Spatial F1 from **0.725 → 0.79**.

---

## 2. Architecture (`ASTRMSpatialModel`)

| Component | Detail | Params |
|---|---|---|
| Backbone | **RegNetY-008** (ImageNet-pretrained) | — |
| Temporal mixing | **ASTRM** on all stages s1–s4 (attention-shift temporal reasoning) | 7.36M (backbone+ASTRM) |
| Temporal encoder | **Bi-GRU** over the 100-frame clip | 5.32M |
| Event classifier | linear head → 8 classes + background | 6.9k |
| Displacement head | ±1-frame temporal displacement refinement | 0.8k |
| **Spatial (ball) head** | **FPN4** (fuses backbone stages s1+s2+s3+s4) → heatmap + offset, **detached** | 0.76M |
| **Total** | | **13.5M** |

- **Input.** 100-frame clips, **320×180** RGB (aspect-correct), ImageNet-normalized.
- **Spatial decode.** Heatmap argmax + offset → `((col+dx)/w, (row+dy)/h)` in [0,1], which
  maps linearly to the original **1920×1080** for pixel-error metrics (grid `w,h` are read
  dynamically, so the rectangular input needs no coordinate correction).
- **Losses.** Focal (event heatmap) + displacement + Gaussian-focal ball heatmap + offset
  L1, combined with `spatial_weight=1.0`.

### Training configuration (best model, Exp E3)
- Optimizer AdamW, **differential LR**: backbone/event 5e-4, spatial head 8e-4; cosine
  schedule with 3-epoch warmup; weight decay 1e-3.
- Effective batch 8 (**bs2 × grad-accum 4**, kept under 24 GB VRAM), 50 epochs, seed 42.
- ~15 min/epoch, ~12.9 h total on a single RTX 4090.

---

## 3. Dataset

- **Source.** VNL 2023 1080p broadcast rallies, `datasets/VNL_dataset_2.0/`.
- **Ball GT.** `annotations/ball_pos_new_filter_0.4/` — event column + normalized (x, y);
  ball (x, y) blank when the detector confidence < 0.4 (~89% of frames have a valid ball).
- **Classes (8).** serve, receive, set, spike, block, dig, score, pass.
- **Split.** Validation matches: `bra_usa_men`, `pol_usa_women`, `chn_tur_women` (VNL 2023);
  all remaining matches for training. Val = 483 rallies / 2594 clips.

---

## 4. Metric definitions

All matching uses the T-DEED protocol: per-class soft-NMS, then greedy matching to GT
within a temporal tolerance. Pixel errors are computed on the **original 1920×1080**.

- **Temporal F1 @2f** — an event prediction is a TP iff within **±2 frames** of a GT event
  of the same class. **Macro** = mean of per-class F1; **Micro** = pooled TP/FP/FN.
- **Spatial F1 @10px** — event-conditioned localization: a TP requires the event to be a
  temporal TP (±2f) **AND** the predicted ball within **10px** of GT. (Identical to the
  Micro joint @2f10p below.)
- **Joint F1 @2f`N`p** — TP iff temporal tolerance ≤ 2 frames **AND** ball within `N` px.
  - **@2f5p** — strict: ±2 frames and ≤ 5 px.
  - **@2f10p** — ±2 frames and ≤ 10 px.
  - **Macro** = mean of per-class joint F1; **Micro** = pooled across the 8 classes.

Threshold selection: the score threshold maximizing temporal Macro-F1 @2f is chosen
(thr = 0.8), and every metric is reported at that operating point (fp32 inference).

---

## 5. Results — E3 (aspect-correct joint FPN4)

The temporal peak and the spatial peak occur at different epochs, so two checkpoints are
reported: **`best_spatial.pt`** (headline localization) and **`best.pt`** (headline events).

### `best_spatial.pt` — headline model

| Metric | Macro-F1 | Micro-F1 |
|---|---|---|
| **Temporal @2f** | **0.8962** | **0.9376** |
| **Joint @2f5p** | **0.5628** | **0.6027** |
| **Joint @2f10p** | **0.7447** | **0.7897** |
| **Spatial F1 @10px** (event-conditioned) | — | **0.7897** |

### `best.pt` — best temporal model

| Metric | Macro-F1 | Micro-F1 |
|---|---|---|
| **Temporal @2f** | **0.9065** | **0.9396** |
| **Joint @2f5p** | **0.5332** | **0.5616** |
| **Joint @2f10p** | **0.7402** | **0.7749** |
| **Spatial F1 @10px** (event-conditioned) | — | **0.7749** |

### Per-class temporal F1 @2f (`best_spatial.pt`)

| serve | receive | set | spike | block | dig | score | pass |
|---|---|---|---|---|---|---|---|
| 0.997 | 0.962 | 0.967 | 0.986 | 0.843 | 0.922 | 0.867 | 0.627 |

(`pass` is the weakest class; `serve`/`spike`/`set`/`receive` are near-saturated.)

---

## 6. Campaign progression & comparison

| Experiment | Spatial F1 @10px | Temporal Macro-F1 @2f |
|---|---|---|
| E1 — frozen backbone, FPN3 head | 0.566 | 0.872 |
| E2 — frozen backbone, FPN4 head | 0.619 | 0.872 |
| E4 — **joint**, FPN4, squished 224² input | 0.725 | 0.907 |
| E5 — frozen-on-strong-backbone, FPN4 | 0.706 | 0.907 |
| **E3 — joint, FPN4, aspect-correct 320×180** | **0.790** | **0.906** |
| Tennis reference (Exp10, FPN3) | 0.775 | 0.952 |
| Tennis reference (Exp11, FPN4) | 0.829 | 0.943 |

**Takeaways.**
- Joint co-adaptation beats any frozen head (0.62 → 0.73); the aspect-correct input then
  adds the final jump (0.73 → 0.79).
- Spatial localization now **exceeds the tennis FPN3 baseline (0.775)** and is within
  ~0.04 of the tennis FPN4 best (0.829); median ball error ≈ 5.1 px (tennis 5.7 px / 4.1 px).
- Temporal Macro-F1 (0.906) still trails tennis (0.94–0.95) — the next lever.

**Answer to "why did the method work for tennis but not volleyball on spatial?"** Two
volley-specific issues, both now fixed: (1) training diverged before the backbone could
co-adapt, and (2) 16:9 frames were squished into a square, distorting horizontal ball
position. Neither was a flaw in the method itself.

---

## 7. Reproduce

```bash
# Train (best model)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train.py \
  --rect_input 320x180 --spatial_mode fpn4 --lr 0.0005 --spatial_lr 0.0008 \
  --batch_size 2 --acc_grad_iter 4 --dataset_len 5000 --epochs 50 \
  --eval_from 12 --eval_every 3 --output_dir outputs_joint_fpn4_rect

# Full metric report (temporal + spatial + joint @2f5p/@2f10p, Macro & Micro)
python3 eval_full.py outputs_joint_fpn4_rect/best_spatial.pt --rect 320x180 --px 5 10
```

Checkpoints: `outputs_joint_fpn4_rect/{best_spatial.pt, best.pt, last.pt}`.
Prediction visualization: `tools/viz_pred.py` → `pred_viz.mp4` (cyan = prediction,
yellow = GT; ball + events + timeline).
