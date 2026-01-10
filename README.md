# LGF-EEG-Emotion

**LGF-EEG-Emotion** is a PyTorch implementation of a **local-global feature fusion framework**
for **subject-independent EEG emotion recognition** under the **Leave-One-Subject-Out (LOSO)**
evaluation protocol.

This repository provides the complete pipeline including EEG feature extraction,
graph-based connectivity modeling, multifractal analysis, model training, and evaluation.

---

## Overview

Cross-subject EEG emotion recognition is challenging due to strong inter-subject variability
and the non-stationary nature of EEG signals.  
This project proposes a **local–global feature fusion framework**, which integrates:

- **Local features**: channel-wise EEG representations
- **Global features**: graph-based connectivity descriptors
- **Auxiliary features**: multifractal characteristics capturing long-range temporal dynamics

A transformer-based architecture is adopted to effectively fuse heterogeneous modalities
and improve subject-independent generalization.

---

## Project Structure
```
LGF-EEG-Emotion/
│
├── datasets/
│ └── dataset_MAET.py
│
├── feature_extraction/
│ ├── extract_EEG_features.py
│ ├── extract_aux25_features.py
│ ├── compute_graph_features_per_trial.py
│ └── multifractal_features.py
│
├── preprocessing/
│ ├── aggregate_npz_to_trial25.py
│ └── generate_labels_csv.py
│
├── models/
│ └── MAET_model.py
│
├── training/
│ ├── train_MAET_LOSO.py
│ └── train_baselines_LOSO.py
│
├── evaluation/
│ ├── compute_loso_mean_confusion_and_metrics.py
│ ├── infer_metrics_from_checkpoints.py
│ ├── subject_accuracy_bar.py
│ └── summarize_baselines.py
│
├── .gitignore
├── LICENSE
└── README.md
```


---

## Dataset

This codebase is designed for **multi-subject EEG emotion datasets**
(e.g., **SEED-VII** or similar benchmarks).

⚠️ **Important Notes**:
- **No raw EEG data, extracted features, labels, or pretrained models are included** in this repository.
- Due to licensing and privacy constraints, users must obtain and preprocess datasets independently.
- Directory paths and file formats should be adapted accordingly.

---

## Feature Extraction

The framework supports multiple complementary feature types:

### Local EEG Features
- Extracted from sliding windows
- Channel-wise representations (e.g., Differential Entropy)

### Global Graph Features
- EEG channels treated as nodes
- Connectivity descriptors computed per trial

### Multifractal Features
- Capture non-linear and long-range temporal dynamics
- Used as auxiliary information to enhance robustness

All window-level features are aggregated into **trial-level representations**
before model training.

---

## Training

### Proposed Model (LOSO)

```bash
python training/train_MAET_LOSO.py
```

### Baseline Models (LOSO)

```bash
python training/train_baselines_LOSO.py
```
Each training run:
- Leaves one subject out for testing
- Trains on all remaining subjects
- Repeats for all subjects

---
## Evaluation

To compute aggregated metrics and confusion matrices:

```bash
python evaluation/compute_loso_mean_confusion_and_metrics.py
```
---
## Dependencies

```bash
pip install -r requirements.txt
```

---

## Reproducibility Guidelines

To ensure fair and reproducible results:
- Fix random seeds for all experiments
- Use consistent window length and step size across subjects
- Apply identical preprocessing and feature extraction pipelines
- Evaluate strictly under the LOSO protocol

---

## Citation

If you find this repository useful in your research, please consider citing it.
A BibTeX entry will be provided upon paper publication.

---

## Acknowledgements

- This implementation was developed as part of an academic research project.

- Some design inspirations are drawn from open-source transformer-based architectures.

---

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.

