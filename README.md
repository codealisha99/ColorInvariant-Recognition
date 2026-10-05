# Color-invariant saree design recognition

Match a saree by the motif on the cloth, including when the same motif is woven in a different palette.

Backbone: torchvision MobileNetV3-Small, started from ImageNet-1K weights (disclosed). The head is a 128-d L2-normalized embedding. Training loss is supervised contrastive. Each photo is one design. Other colorways are Lab chroma rotations of that photo, because the DeepLure corpus does not label real colorways. One view in each batch shares a hue across designs, so different motifs in the same palette are negatives. At test time the gallery crop is the center of the field and the query crop is shifted and mirrored, at a disjoint hue.

## Submit

The Kaggle notebook is `notebooks/deeplure_saree.ipynb`. It already contains a finished run. Upload it to Kaggle, set Internet On, GPU, and Public, then Save & Run All. Put the public notebook URL in https://forms.gle/1fJaAFbuTS4FdhQn9. Do not upload `data/`.

`notebooks/submission.ipynb` is the repo walkthrough. It calls these modules, loads the checkpoints, and trains only if a checkpoint is missing.

## Kaggle

Public notebook (GPU, Internet On, Save & Run All):

> Kaggle notebook URL: _TODO — paste the public notebook link here after Save & Run All_

The notebook is self-contained: it writes `src/` into the session, downloads the Drive corpus at runtime, trains 40 epochs, and prints fresh identification, verification, and efficiency numbers.

## Setup

On this machine the environment, the 165 photos, and the trained checkpoint are already in place. From the project folder:

```bash
cd deeplure-Alishakarma
source .venv/bin/activate
python -m src.infer --query data/sarees/h_img_34590.jpg --topk 5
```

That scores one saree against the local gallery using `runs/color/best.pt`.

To set it up from scratch on another machine:

```bash
cd deeplure-Alishakarma
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.download
python -m src.train --data data/sarees --out runs/color
python -m src.eval --checkpoint runs/color/best.pt
```

`python -m src.download` pulls the Drive corpus into `data/sarees`. That folder is proprietary: do not commit it or upload it. Training is only required when `runs/color/best.pt` is missing. Open `notebooks/submission.ipynb` from the project root so `src/` imports.

The rest of the checks:

```bash
python -m src.train --data data/sarees --out runs/nocolor --no-color-aug
python -m src.eval --checkpoint runs/nocolor/best.pt
python -m src.eval --imagenet          # frozen ImageNet backbone, same split
python -m src.efficiency --checkpoint runs/color/best.pt
```

`--no-color-aug` trains the ablation that is allowed to use color. That run is in `runs/nocolor`.

Held-out test, 33 designs, synthetic palettes: color-trained model Rank-1 90.9%, Rank-5 100%, mAP 95.5%, verification ROC-AUC 0.992, EER 3.0%. Same-palette impostors still Rank-1 90.9%. The ablation without palette changes is Rank-1 63.6% on the same split. A frozen ImageNet backbone, with no fine-tuning, is also Rank-1 66.7%, with ROC-AUC 0.944 and EER 15.2%. Encoder is 1.0M parameters, 0.11 GFLOPs, 128-d, 2–5 ms per image on Apple GPU (43 ms on CPU).

Do not commit or redistribute `data/`. Delete it when the exercise is done.

## Approach note

444 characters. The form limit is 500.

MobileNetV3-Small maps a 224px crop to a 128-d L2-normalized embedding. Each photo is one design. Positives are its crops under Lab hue rotation and saturation change; other designs in the batch are negatives. Supervised contrastive loss, PK sampling, flip and mild blur. Cosine similarity ranks the gallery and verifies pairs. ImageNet initialization, disclosed. Real colorway labels are absent, so palettes are synthetic and the eval says so.
