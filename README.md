# Color-Invariant Saree Design Recognition

Match a saree by the motif on the cloth, even when the same motif is woven in a different palette.

## 1. Project Overview

This repository contains a PyTorch system that identifies saree designs from RGB photos. Given one query photo, it ranks a gallery of known designs by similarity (identification) and can decide whether any two photos show the same design (verification). The defining requirement is **color invariance**: the same motif in different palettes must match, while different motifs in the same palette must not.

## 2. Problem Statement

- Input: an RGB image of a saree.
- Gallery: images of known saree designs (165 photos in the supplied corpus).
- Output A (identification): gallery ranked by similarity, best match(es) returned.
- Output B (verification): same-design / different-design decision for an image pair.

## 3. Core Mindset

This is framed as an **image-retrieval / metric-learning problem, not fixed-class classification**. Classification asks "which fixed class is this?" — useless for unseen designs. Metric learning asks "how close is this design to another design?" Each photo is treated as its own design identity, the network maps crops to points in a 128-D space, and cosine similarity does both ranking and verification. New designs can be added to the gallery without retraining.

## 4. Dataset

- `data/sarees/`: 165 JPEGs, flat layout (e.g. `h_img_34590.jpg`). Proprietary — gitignored, never committed.
- Split by photo, seed 42: **train 99 / validation 33 / test 33** (`src/data.py:split_designs`). A crop or recoloring of one file never sits on opposite sides of the split.
- Filenames carry no design id and no colorway label; a sibling `normal_sarees` folder was empty, and the Kaggle weave-category set (Ikat/Banarasi/…) labels categories, not designs, so it is not used.

Because the supplied corpus does not contain design-level colourway annotations, synthetic Lab chroma transformations are used as a controlled proxy for alternate colourways.

The evaluation therefore measures robustness to controlled colour transformations rather than proving generalization to independently photographed real-world colourways.

## 5. Approach Note

444 characters. Form limit is 500.

```text
MobileNetV3-Small maps a 224px crop to a 128-d L2-normalized embedding. Each photo is one design. Positives are its crops under Lab hue rotation and saturation change; other designs in the batch are negatives. Supervised contrastive loss, PK sampling, flip and mild blur. Cosine similarity ranks the gallery and verifies pairs. ImageNet initialization, disclosed. Real colorway labels are absent, so palettes are synthetic and the eval says so.
```

## 6. High-Level Architecture

```text
Data → Preprocessing → Augmentation → Backbone → Projection head → Embedding → Retrieval / Verification
```

## 7. HLD

```text
Source photo
  └─ Split by photo, seed 42 (99 train / 33 val / 33 test)
       ├─ TRAIN: random crop + Lab hue rotation + flip/blur → P×K batch
       │         → MobileNetV3-Small → 576-d → 128-d + BN → L2 norm
       │         → supervised contrastive loss → AdamW + cosine schedule
       │         → validation Rank-1 selects runs/color/best.pt
       └─ EVAL: gallery (center crop, hue +40°) vs query (shifted + mirrored crop, hue +160°)
                → cosine similarity → Rank-k / ROC / thresholded verification
```

## 8. Low-Level Design

- `src/color.py` — sRGB↔XYZ↔Lab conversion, chroma-plane hue rotation, saturation scaling, palette matching.
- `src/data.py` — photo-level split, crop regions, P×K batch sampling, training view generation.
- `src/model.py` — `SareeEncoder`: MobileNetV3-Small trunk + 128-d head; `forward_backbone` exposes pooled features for the frozen baseline.
- `src/loss.py` — supervised contrastive loss (temperature-scaled, self-comparison masked).
- `src/train.py` — AdamW loop, cosine schedule, per-epoch validation, `best.pt`/`last.pt` + `history.json`.
- `src/eval.py` — cross-palette identification, verification, same-palette stress test, ImageNet baseline (`--imagenet`).
- `src/infer.py` — one query scored against a gallery folder, top-K printed.
- `src/efficiency.py` — parameter count, Conv/Linear FLOPs, single-image latency → `efficiency.json`.
- `src/download.py` — pulls the Drive corpus into `data/sarees`.
- `src/clustering/` — isolated experimental unsupervised analysis (see §32).

## 9. Model Architecture

```text
RGB image → 224×224 crop → MobileNetV3-Small → global average pool (576-d)
  → Linear 576→128 (no bias) → BatchNorm → L2 normalize → 128-d embedding
```

Backbone initialized from ImageNet-1K weights (torchvision `MobileNet_V3_Small_Weights.DEFAULT`, disclosed); the classifier is discarded and the whole trunk is fine-tuned. Small backbone chosen deliberately: the retrieval quality saturates while parameters, FLOPs, and latency stay low.

## 10. Preprocessing Pipeline

Eval/infer path: resize shorter side (360 px), take the 224 crop, ImageNet-normalize. Gallery uses the center crop; the query crop is shifted +48 px and mirrored so pixel copies cannot match. Train path: `RandomResizedCrop(224, scale 0.55–1.0)`, horizontal flip p=0.5, Gaussian blur p=0.2. No vertical flip or 90° rotations — motif orientation is semantic.

## 11. Lab Color-Invariance Strategy

```text
RGB → XYZ → Lab → rotate the chromatic (a, b) plane → scale saturation → back to RGB
```

Only chroma changes; lightness (hence motif edges and geometry) is preserved. Grayscale was rejected: many motifs are hue contrast on one ground, which gray would erase.

## 12. Synthetic Colorway Generation

Each source photo is one design. Alternate "colorways" are full-circle Lab hue rotations with saturation scaled 0.6–1.4; ~15% of views keep a near-original hue (±20°) so the source palette is still seen. Train and eval palettes are disjoint (gallery +40°, query +160°).

## 13. Hard Same-Palette Negatives

View 0 of **every** design in a batch shares one random hue (`src/data.py:143`), so the batch contains different motifs rendered in the same palette. Same color therefore cannot become the identity signal: "same color does not imply same design." This is the key anti-shortcut mechanism.

## 14. P×K Sampling

P = 16 designs × K = 4 views = 64 crops per batch. Views of the same design are positives (including its recolorings); every view of another design is a negative.

## 15. Supervised Contrastive Loss

Temperature 0.07. Same-design views are pulled together, different-design views pushed apart, with the self-comparison masked out of the denominator. Conceptually: positives attract, negatives repel, scaled by temperature.

## 16. Training Configuration

| Setting | Value |
|---|---|
| Backbone / init | MobileNetV3-Small / ImageNet-1K |
| Input / embedding | 224×224 / 128-d L2-normalized |
| P / K / batch | 16 / 4 / 64 |
| Loss / temperature | Supervised contrastive / 0.07 |
| Optimizer / LR / decay | AdamW / 3e-4 / 1e-4 |
| Scheduler / epochs / seed | Cosine / 40 / 42 |
| Augmentation | Lab hue + sat 0.6–1.4, flip p=0.5, blur p=0.2 |

## 17. Training Flow

Dataset → photo split → P×K sampling → multiple views per design → random crop → color augmentation → flip/blur → MobileNetV3-Small → 576-d → 128-d projection → L2 norm → SupCon loss → backprop (AdamW, cosine) → per-epoch validation → best checkpoint.

Validate with `runs/color/history.json` (40 epochs; best val Rank-1 96.97%).

## 18. Inference Flow

Query image → center-crop preprocessing → 128-d normalized embedding → dot product against gallery embeddings → sort → top-K. No hue rotation at inference: the photo is scored as it is.

## 19. Retrieval

Score = cosine(query, gallery) = dot product (embeddings are unit length). The gallery is ~165 images, so direct vectorized comparison (under a second) is sufficient; FAISS is **not** used and no ANN index is claimed. FAISS/HNSW would be the production answer for hundreds of thousands of designs.

## 20. Verification

Cosine similarity against the validation-selected threshold **0.67** (`runs/color/eval.json`: `val_eer_threshold 0.6699…`). Above it: same design. The threshold is fit on validation EER and applied unchanged to test — test data never tunes it, avoiding evaluation leakage.

## 21. Evaluation Protocol

- **Identification** (one relevant gallery item per query): Rank-1 and Rank-5 (fraction of queries whose match is in the top 1 / 5 — i.e. Recall@1/@5), mAP (mean of 1/rank).
- **Verification** (33 positives + 1056 negatives): ROC-AUC (threshold-free ranking quality), EER (error at the FAR=FRR point), TAR@FAR=1% (recall at a strict operating point), plus TAR/FAR/Accuracy/Precision/Recall/F1 at the deployed 0.67 threshold.
- **Stress test**: every gallery image recolored to the query's palette; Rank-1 must then come from motif, not color.

## 22. Main Results

Test, 33 held-out designs (`runs/color/eval.json`, re-run verified bit-identical):

| | Rank-1 | Rank-5 | mAP | ROC-AUC | EER |
|---|---|---|---|---|---|
| Cross-palette identification / verification | **90.91%** | 100% | 95.45% | 99.19% | 3.03% |
| Same-palette stress | 90.91% | 100% | 94.95% | — | — |

Validation (for reference): Rank-1 96.97%, mAP 98.48%, ROC-AUC 99.63%, EER 2.37%.

## 23. Verification Results

At the unchanged validation threshold 0.67 applied to test: **TAR 96.97%, FAR 4.17%**; TAR@FAR=1% is 75.76% (a stricter operating point, not the deployed one).

## 24. Original Query Results

Query with its original palette (shifted/mirrored crop, no recolor): Rank-1 87.88%, Rank-5 100%, mAP 93.94% — slightly below the recolored-query number, as expected when geometry shifts without palette change.

## 25. Same-Palette Stress Test

All gallery images palette-matched to each query: Rank-1 90.91%, mAP 94.95%. The model still retrieves the motif when color is equalized — direct evidence against color shortcutting.

## 26. Color-Invariance Sanity Check

Same photo before/after a +160° Lab hue rotation (identical center crops): cosine **0.983**. Under the shifted eval crops the same recolor scores ≈0.92–0.93. This is a sanity check, not the main metric — and invariance is not perfect (see §28).

## 27. Controlled Similarity Analysis

Mean cosine over the test split (re-measured):

| | Same palette | Different palette |
|---|---|---|
| Same design | 0.872 | 0.849 |
| Different design | 0.029 | 0.024 |

Changing the palette of one design moves the embedding far less than changing the design itself. All 33 same-design/different-palette scores sit far above the impostor mean. This shows robustness to controlled color change, not proof of real-world colorway generalization.

## 28. Hard Cases

- Hardest same-palette impostor observed: ≈0.93 cosine between visually similar dense-gold-jaal designs. Without motif-level labels this cannot be certified a false positive — it may be the same design family photographed twice.
- Hardest same-image recolor: 0.634 (`img_65263.jpg`, eval geometry) — above the 0.67 threshold? No: below it, so this true pair would be rejected at the operating point. Perfect invariance is not claimed.

## 29. Ablation: No Color Augmentation

`runs/nocolor` (same code, `--no-color-aug`; `eval.json` regenerated from the current checkpoint):

| | Rank-1 | Rank-5 | mAP | ROC-AUC | EER |
|---|---|---|---|---|---|
| Color augmentation | 90.91% | 100% | 95.45% | 99.19% | 3.03% |
| No color augmentation | 63.64% | 90.91% | 77.71% | 96.84% | 9.09% |

Rank-1 improves by **+27.27 percentage points** (90.91 − 63.64); EER falls 9.09% → 3.03%. This is the strongest evidence the color-invariance training works. (An older stale `nocolor/eval.json` with different numbers was replaced; report only the fresh values above.)

## 30. Ablation: Frozen ImageNet

`python -m src.eval --imagenet` — pooled trunk features, no fine-tuning, no projection (`runs/imagenet/eval.json`):

| | Rank-1 | Rank-5 | mAP | ROC-AUC | EER |
|---|---|---|---|---|---|
| Frozen ImageNet | 66.7% | 97.0% | 80.4% | 0.944 | 15.2% |

Fine-tuned metric learning (90.91%) beats generic ImageNet features (66.7%): task-specific training matters.

## 31. Efficiency

Measured (`runs/color/efficiency.json`, CPU re-measured locally):

```text
~1.0M parameters (1,000,992)
128-D embedding (512 bytes FP32)
~0.11 GFLOPs (Conv/Linear MACs)
~4.13 MB checkpoint
~5 ms/image Apple Silicon MPS (per artifact; observed range 2–5 ms)
~35–45 ms/image CPU (35.5 ms measured here; hardware-dependent)
```

No T4/Kaggle GPU latency is claimed — it was not measured. Latency depends on hardware, batch size, warm-up, and PyTorch version.

## 32. Exploratory Clustering

Optional, isolated module (`src/clustering/`, outputs under `runs/clustering/` — delete both to remove it). Cosine DBSCAN (eps 0.25, min_samples 3) on the frozen 128-d embeddings: **22 clusters, 34 noise, 131 assigned**; silhouette 0.381, Davies–Bouldin 1.105. Artifacts: `embedding_map.png`, `contact_sheets/`, `cluster_summary.csv`, `clusters.json`, `representatives.json`.

Call this only "exploratory unsupervised embedding analysis" — there are no ground-truth motif labels, so there is no clustering accuracy and no "Motif A/B" identities.

## 33. Example Inference

Qualitative top-1 hits (re-run verified; distinct from aggregate metrics):

```text
img_357974.jpg → img_94825.jpg  = 0.951
img_314642.jpg → img_649782.jpg = 0.864
img_120749.jpg → img_399807.jpg = 0.723
img_987571.jpg → img_987632.jpg = 0.926
img_413996.jpg → img_648754.jpg = 0.895
```

## 34. Project Structure

```text
deeplure-Alishakarma/
├── README.md
├── architecture.md
├── requirements.txt
├── brief/
├── notebooks/
│   ├── deeplure_saree.ipynb   # primary Kaggle submission notebook (37 cells, executed)
│   └── submission.ipynb       # local repo walkthrough
├── data/
│   └── sarees/                # 165 JPEGs, proprietary, gitignored
├── runs/
│   ├── color/                 # best.pt, eval.json, efficiency.json, history.json
│   ├── nocolor/               # ablation checkpoint + eval
│   ├── imagenet/              # frozen-baseline eval
│   ├── analysis/              # verification ROC / score-distribution figures
│   └── clustering/            # exploratory clustering outputs
└── src/
    ├── color.py
    ├── data.py
    ├── model.py
    ├── loss.py
    ├── train.py
    ├── eval.py
    ├── infer.py
    ├── efficiency.py
    ├── download.py
    └── clustering/
```

## 35. Important Modules

See §8. `src/model.py` defines `SareeEncoder` (trunk + 128-d head) and `forward_backbone` used only by the frozen baseline; `src/eval.py` implements all three test protocols plus `--imagenet`; `src/infer.py` needs only `--query` (gallery defaults to `data/sarees`).

## 36. Setup

```bash
cd deeplure-Alishakarma
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 37. Dataset Download

```bash
python -m src.download
```

Pulls the Drive corpus into `data/sarees` (default `--dest`). Requires internet. Proprietary: never commit or redistribute.

## 38. Training Commands

```bash
python -m src.train --data data/sarees --out runs/color
```

Defaults (match `src/train.py` argparse): 40 epochs, P=16, K=4, 128-d, lr 3e-4, temperature 0.07, seed 42. Only needed when `runs/color/best.pt` is missing.

## 39. Evaluation Commands

```bash
python -m src.eval --checkpoint runs/color/best.pt
python -m src.eval --checkpoint runs/nocolor/best.pt
python -m src.eval --imagenet
```

Each writes `eval.json` next to the checkpoint (or `--out PATH`).

## 40. Inference Commands

```bash
python -m src.infer --query data/sarees/h_img_34590.jpg --topk 5
```

`--gallery` defaults to `data/sarees`; `--checkpoint` defaults to `runs/color/best.pt`.

## 41. Ablation Commands

```bash
python -m src.train --data data/sarees --out runs/nocolor --no-color-aug
python -m src.eval --checkpoint runs/nocolor/best.pt
```

## 42. Efficiency Command

```bash
python -m src.efficiency --checkpoint runs/color/best.pt
```

## 43. Kaggle Submission

`notebooks/deeplure_saree.ipynb` (37 cells, 19 code, 16 with stored outputs) is the primary submission notebook: environment setup → authorized dataset download → source reconstruction → 40-epoch training → identification + verification eval → retrieval strips → hue-sweep demo → verification detail with ROC → efficiency. `notebooks/submission.ipynb` is the local walkthrough. Kaggle settings: GPU on, Internet on, notebook Public, then Save & Run All; paste the URL into the assignment form.

> Kaggle notebook URL: _TODO — paste the public notebook link here after Save & Run All_

Requires final Kaggle Save & Run All verification — remote execution has not been confirmed from this environment.

## 44. Reproducibility

Seed 42 everywhere (split, sampling, init); derived views never cross splits; repeated local evals verified bit-identical; checkpoint loads on CPU with embedding shape `(2, 128)` and norms ≈1.0; device fallback CUDA → MPS → CPU. Environment: Python 3.12, torch 2.14.1, torchvision 0.29.1, numpy 2.5.3, pillow 12.3.0.

## 45. Data Privacy

`data/sarees/` is proprietary and gitignored. The public repository contains code, docs, and permitted artifacts only — no images. Delete the local corpus when the exercise is done.

## 46. Limitations

1. Synthetic Lab recolorings are not real photographed alternate colourways.
2. Small corpus (165 photos); some visually similar designs remain hard (≈0.93 impostor, 0.634 true-pair stress case).
3. No motif-level ground truth, so error analysis is qualitative.
4. 165-image gallery needs no ANN index; production scale would need FAISS/HNSW.
5. Kaggle reproducibility pending remote execution; no T4 latency claimed.

## 47. Future Work

Real multi-colourway pairs with design IDs; explicit hard-negative mining; multi-scale and border/pallu-aware crops; lightweight ViT/ConvNeXt comparison; illumination robustness; larger external benchmark; FAISS retrieval; calibrated verification thresholds; human-reviewed errors; exact Kaggle GPU benchmarking.

## 48. STAR Summary

- **Situation:** design-level saree identification independent of color, but the corpus lacked real colourway labels.
- **Task:** working PyTorch retrieval + verification focused on motif structure, not palette.
- **Action:** metric-learning formulation; MobileNetV3-Small → 128-d L2 embedding; SupCon with P×K sampling; Lab hue/saturation augmentation with same-palette hard negatives; retrieval/verification/stress evals vs no-color and frozen-ImageNet baselines.
- **Result:** 90.91% Rank-1, 100% Rank-5, 95.45% mAP, 99.19% ROC-AUC, 3.03% EER; color augmentation alone added +27.27pp Rank-1.

## 49. Interview Explanation

"I used MobileNetV3-Small as the pretrained visual backbone. Its 576-dimensional pooled representation goes through a 576-to-128 projection head with BatchNorm and L2 normalization to produce the retrieval embedding. I fine-tuned this using supervised contrastive learning with P×K sampling and deliberately introduced Lab-based color transformations and same-palette hard negatives to make the embedding invariant to color while preserving motif differences. At inference, I use cosine similarity for gallery retrieval and validation-selected thresholding for pair verification."

## 50. Final Results Summary

| Experiment | Rank-1 | Rank-5 | mAP | ROC-AUC | EER |
|---|---:|---:|---:|---:|---:|
| Main color-invariant model | 90.91% | 100% | 95.45% | 99.19% | 3.03% |
| No color augmentation | 63.64% | 90.91% | 77.71% | 96.84% | 9.09% |
| Frozen ImageNet | 66.7% | 97.0% | 80.4% | 94.4% | 15.2% |

```text
Main model:
~1.0M parameters
128-D embedding
~0.11 GFLOPs
~4.13 MB checkpoint
~35–45 ms CPU latency (35.5 ms measured here; hardware-dependent)
~2–5 ms Apple GPU/MPS latency
```

## 51. Final Takeaway

Color is a shortcut; the training distribution removes it. Same-palette hard negatives plus disjoint-hue evaluation prove the embedding tracks motif geometry, with honest, measured limits.
