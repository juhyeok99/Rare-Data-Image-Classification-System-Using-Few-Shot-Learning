# Rare Data Image Classification System Using Few-Shot Learning

Implementation of the system proposed in:

> **Rare Data Image Classification System Using Few-Shot Learning**  
> Juhyeok Lee and Mihui Kim  
> *Electronics* 2024, 13, 3923. https://doi.org/10.3390/electronics13193923

---

## Overview

Standard deep learning classifiers struggle when training data is scarce.
This project tackles that by combining two things:

1. **Structure-aware data augmentation** — an encoder-decoder that blends style
   information from a reference image into a source image while preserving the
   source object structure (via Sobel-based structural features). Generated
   images are filtered by PSNR and SSIM before being added to the dataset.

2. **Few-shot learning classifier** — a CNN backbone that extracts feature
   vectors, which are then classified by cosine similarity to per-class
   prototypes (Equation 1 from the paper). The prototype head can adapt to
   new data without retraining.

Compared to a plain CNN baseline, the proposed system improves classification
accuracy by roughly 15–16% when only 100 samples are available.

---

## Project Structure

```
few-shot-rare-data/
├── config.py                   # all hyperparameters in one place
├── prepare_data.py             # train augmentor, generate fake images
├── run_experiments.py          # reproduce paper experiments
├── inference.py                # classify new images with a trained model
│
├── data/
│   ├── augmentation.py         # StructureAwareAugmentor (proposed)
│   ├── dataset.py              # RareImageDataset + FewShotEpisodeSampler
│   └── preprocessing.py        # PSNR, SSIM, quality_filter
│
├── models/
│   ├── cnn_baseline.py         # CNN comparison model
│   └── proposed_system.py      # ProposedFewShotSystem (CNN backbone + FSL head)
│
├── experiments/
│   ├── train_cnn.py            # CNN training loop
│   ├── train_proposed.py       # episodic FSL training loop
│   └── bayesian_opt.py         # Bayesian hyperparameter search
│
├── utils/
│   ├── metrics.py              # accuracy / precision / recall / F1
│   ├── visualization.py        # bar charts, accuracy curves
│   └── checkpoint.py           # save / load model weights
│
├── tests/
│   └── test_sanity.py          # quick sanity checks for all components
│
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/your-username/few-shot-rare-data.git
cd few-shot-rare-data
pip install -r requirements.txt
```

Python 3.9+ recommended.

---

## Data Preparation

The paper uses the [WikiArt dataset](https://www.kaggle.com/datasets/ipythonx/wikiart-gangogh-creating-art-gan).
Organise your raw images like this:

```
data/raw/
├── paintings/   # real artwork images
└── plants/      # real plant images (web-crawled)

data/artgan/
├── paintings/   # ArtGAN-generated images from WikiArt
└── plants/
```

Then train the augmentation model and generate the fake dataset:

```bash
# paintings
python prepare_data.py \
    --raw_dir  data/raw/paintings \
    --out_dir  data/augmented/paintings \
    --aug_ckpt checkpoints/augmentor_painting.pt \
    --epochs   50

# plants
python prepare_data.py \
    --raw_dir  data/raw/plants \
    --out_dir  data/augmented/plants \
    --aug_ckpt checkpoints/augmentor_plant.pt \
    --epochs   50
```

The augmentor runs PSNR ≥ 20 dB and SSIM ≥ 0.5 quality gates before saving.
Low-quality generated images are silently dropped.

---

## Running Experiments

### Reproduce Table 1 & Figure 7 (proposed system augmentation, paintings)

```bash
python run_experiments.py \
    --real_dir     data/raw/paintings \
    --fake_dir     data/augmented/paintings \
    --dataset_name painting_proposed \
    --data_sizes   1000 500 100 \
    --epochs       1000
```

### Reproduce Figure 6 (ArtGAN, paintings)

```bash
python run_experiments.py \
    --real_dir     data/raw/paintings \
    --fake_dir     data/artgan/paintings \
    --dataset_name painting_artgan \
    --data_sizes   1000 500 100 \
    --epochs       1000
```

### Plant experiments (Figures 8 & 9 / Table 2)

```bash
# proposed system augmentation
python run_experiments.py \
    --real_dir     data/raw/plants \
    --fake_dir     data/augmented/plants \
    --dataset_name plant_proposed \
    --data_sizes   1000 500 100

# ArtGAN
python run_experiments.py \
    --real_dir     data/raw/plants \
    --fake_dir     data/artgan/plants \
    --dataset_name plant_artgan \
    --data_sizes   1000 500 100
```

Add `--bayes_opt` to run Bayesian hyperparameter search before training.
Results (JSON + charts) are saved under `results/<dataset_name>/`.

---

## Inference

```bash
# Proposed system (needs a small support set)
python inference.py \
    --model        proposed \
    --ckpt         checkpoints/proposed_painting_n100.pt \
    --support_real data/raw/paintings \
    --support_fake data/augmented/paintings \
    --query_dir    path/to/new_images \
    --n_support    5

# CNN baseline
python inference.py \
    --model     cnn \
    --ckpt      checkpoints/cnn_painting_n100.pt \
    --query_dir path/to/new_images
```

---

## Sanity Tests

```bash
python tests/test_sanity.py
```

Checks every model component (encoder, structural extractor, fusion module,
decoder, prototype head, full system) on random dummy tensors.

---

## Key Hyperparameters

| Parameter      | Value  | Source                    |
|----------------|--------|---------------------------|
| Epochs         | 1000   | Paper Section 4.1         |
| Batch size     | 32     | Paper Section 4.1         |
| Learning rate  | 0.001  | Bayesian optimisation [21]|
| Dropout        | 0.5    | Paper Section 4.1         |
| N-way          | 2      | Binary: real vs. fake     |
| N-support      | 5      | Default few-shot setting  |
| PSNR threshold | 20 dB  | Paper Section 3.1 [12]    |
| SSIM threshold | 0.5    | Paper Section 3.1 [13]    |

All hyperparameters are in `config.py`.

---

## Results

Classification accuracy on 100-sample subsets (from paper Tables 1 & 2):

**Painting data (proposed augmentation)**

| Method          | Accuracy | Recall | Precision | F1     |
|-----------------|----------|--------|-----------|--------|
| CNN             | 0.5876   | 0.5741 | 0.5803    | 0.5978 |
| Proposed system | 0.6781   | 0.6812 | 0.6934    | 0.7196 |

**Plant data (proposed augmentation)**

| Method          | Accuracy | Recall | Precision | F1     |
|-----------------|----------|--------|-----------|--------|
| CNN             | 0.5881   | 0.5614 | 0.5716    | 0.5831 |
| Proposed system | 0.5876   | 0.6763 | 0.6901    | 0.7089 |

---

## Citation

```bibtex
@article{lee2024rare,
  title     = {Rare Data Image Classification System Using Few-Shot Learning},
  author    = {Lee, Juhyeok and Kim, Mihui},
  journal   = {Electronics},
  volume    = {13},
  number    = {19},
  pages     = {3923},
  year      = {2024},
  publisher = {MDPI},
  doi       = {10.3390/electronics13193923}
}
```

---

## License

MIT
