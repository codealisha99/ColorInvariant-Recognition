# DeepLure AIE-CASE — build plan

Saved source: `brief/DeepLure_JD.pdf`
Deadline: **11:59 PM IST, 5 Oct 2026**
Submit: https://forms.gle/1fJaAFbuTS4FdhQn9 (public notebook or repo link)

## What the task actually is

Face recognition for saree motifs.

- Input: one RGB saree image.
- Gallery: known designs. The same motif may exist in several colorways.
- Identification: rank gallery by similarity, return best match(es).
- Verification: given two images, say whether they are the same design.
- Hard constraint: same motif in a different palette must match. A different motif in the same palette must not.

Sarees only for the scored result. The embedding should still be a generic textile encoder so a later garment head is a data change, not an architecture change.

## What the data actually is

Drive folder `sarees_dataset` (do not commit, do not redistribute, delete after submission):

- `handloom_sarees`: 165 JPEGs, publicly listed. A sample is a flat cream ground with a repeating purple floral and a gold geometric motif, plus a sparser border. No design-id in the filename.
- `normal_sarees`: sibling folder. Embedded listing returned no files; size still unknown. This matches the public Neural Loom layout (normal vs handloom crops, often 256×256).

Kaggle `div456/indian-saree-patterns`: 1000+ images, **4 weave classes** (Ikat, Banarasi, Pichwai, Bandhani). That is category classification, not “design 42 in red and in blue.”

Neither source labels colorways of one design. We have to create that supervision.

## How to build it

### Representation

Pretrained **MobileNetV3-Small** → 128-d L2-normalized embedding. Small enough for the Kaggle free GPU, and the efficiency report (params, FLOPs, latency, embedding size) is bonus credit.

Do not drop color and train on grayscale. Many motifs are close in luminance and exist as hue contrast (the purple flower vs the gold flower on the same cream ground). Grayscale would erase the thing we are trying to recognize.

Color invariance comes from the **training distribution**, not from throwing chroma away:

- Each source photo is one design id.
- Positives are crops of that photo plus recolorings: full hue rotation, saturation jitter, and a palette remap that keeps spatial layout and luminance edges.
- Hard negatives are **different designs recolored into the query palette**, so shared color cannot be the shortcut.

### Training

- Loss: supervised contrastive (SupCon). Design id is the class.
- Sampler: P designs × K colorways per batch.
- Augment: resized crop, horizontal flip, mild blur, the color pipeline above.
- Pretrained ImageNet backbone, disclosed. Freeze early layers for a few epochs, then fine-tune.
- Split by source image, never by crop. Crops of one photo stay in the same split.

### Inference

- Embedding is the backbone pooled vector, linearly projected, L2-normalized.
- Identification: cosine similarity against the gallery, return top-k.
- Verification: cosine score against a threshold chosen on validation (equal-error or max Youden).
- Optional and cheap: average two or three hue-jittered views of the query at test time.

### Pre / post pipeline (for the 500-character note)

Resize shorter side, random resized crop to 224 at train / center crop at test, ImageNet normalize. Color pipeline only at train. Post: L2 normalize, cosine rank, threshold.

### Evaluation (this is what they grade)

Hold out designs, not random images.

- Gallery: one or more colorways per design.
- Query: a held-out colorway of those designs, palette disjoint from its gallery mate.
- Identification: Rank-1, Rank-5, mAP.
- Verification: ROC-AUC, EER, TAR at FAR = 1%.
- Stress test: impostors recolored to the query palette. Rank-1 must still be the true motif.
- Ablation: same model trained without palette remap, to show the invariance is learned.

Report that real multi-colorway labels were not in the release, so paired colorways are synthetic remaps of held-out photos, with the protocol written down.

### 500-character approach note (draft)

MobileNetV3-Small maps a 224px crop to a 128-d L2-normalized embedding. Each photo is one design. Positives are its crops under hue rotation, saturation jitter, and palette remap; hard negatives are other motifs recolored into the query palette. Supervised contrastive loss, PK sampling, flip and mild blur. Cosine similarity ranks the gallery and a validation threshold verifies pairs. Pretrained ImageNet weights, disclosed.

## Repo shape (when we build)

```
src/data.py        # download helpers, design ids, synthetic colorways, splits
src/model.py       # MobileNetV3-Small + projection
src/train.py       # SupCon loop
src/eval.py        # identification + verification + stress test
src/efficiency.py  # params, FLOPs, latency
notebooks/submission.ipynb   # runnable top to bottom
data/              # gitignored, proprietary
```

PyTorch only. No proprietary images in git.
