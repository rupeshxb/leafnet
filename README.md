# LeafNet — CNN-Based Potato Leaf Disease Classification

A comparative deep learning study evaluating four CNN architectures — VGG16, ResNet50, MobileNetV2, and a custom CNN — for automated classification of potato (*Solanum tuberosum* L.) leaf disease from a single photograph, deployed as a zero-cost web application for smallholder farmers.

Developed as part of an IT Project thesis (MIT programme) at Himalayan College of Management, affiliated with IUKL.

## Overview

Potato cultivation is highly vulnerable to foliar disease, and manual field inspection is slow, subjective, and inaccessible to many smallholder farming communities in Nepal. This project trains and benchmarks four CNN architectures on the PlantVillage dataset to classify potato leaves as **Early Blight**, **Late Blight**, or **Healthy**, then deploys the best-performing model behind a simple web interface that returns a diagnosis, confidence score, and recommended action from a single uploaded photo.

The original research scope targeted five disease classes; this was narrowed to three (Early Blight, Late Blight, Healthy) during dataset compilation, since no sufficiently large, quality-verified, publicly available corpus for Bacterial Wilt or Septoria Leaf Spot in potato foliage could be located. See the thesis Chapter 1 (§1.6) for the full rationale.

## Dataset

- **Source**: PlantVillage potato subset, accessed via Kaggle ([arjuntejaswi/plant-village](https://www.kaggle.com/datasets/arjuntejaswi/plant-village))
- **Licence**: Creative Commons Attribution 4.0 (CC BY 4.0) — attribution provided via citation of Mohanty et al. (2016) below
- **Classes**: Early Blight (1,000 images), Late Blight (1,000 images), Healthy (152 images) — 2,152 images total
- **Split**: Stratified 70/15/15 (1,506 train / 323 val / 323 test)

## Architectures Compared

| Model | Approach | Parameters (approx.) |
|---|---|---|
| VGG16 | Transfer learning, ImageNet pretrained | 138M |
| ResNet50 | Transfer learning, ImageNet pretrained | 25.6M |
| MobileNetV2 | Transfer learning, ImageNet pretrained | 3.5M |
| Custom CNN | Trained from scratch | — |

All three transfer learning models followed a two-phase protocol: frozen-base feature extraction, then fine-tuning of the last 20 base layers at a reduced learning rate. Full configuration in `notebook/LeafNet_Training.ipynb` and Chapter 3 of the thesis.

## Results

Evaluated on a held-out test set (323 images) within a single training and evaluation session.

| Model | Accuracy | Macro F1-Score | Avg. Inference (ms) |
|---|---|---|---|
| VGG16 | 95.36% | 0.9201 | 669.16 |
| ResNet50 | 82.04% | 0.5674 | 222.39 |
| **MobileNetV2** | **99.07%** | **0.9873** | **94.46** |
| Custom CNN | 97.52% | 0.9550 | 120.98 |

**MobileNetV2** was selected for deployment, achieving the highest accuracy, the highest macro F1-score, and the fastest inference of the four architectures — directly relevant for deployment on resource-constrained devices.

**Notable finding**: ResNet50 achieved 0.00 precision, recall, and F1-score on the Healthy class despite a respectable headline accuracy of 82.04% — a complete failure to ever correctly classify a healthy leaf, masked by aggregate accuracy alone. This is discussed in detail in Chapter 4 of the thesis as direct evidence for why macro-averaged F1, not raw accuracy, was adopted as the primary evaluation metric.

### A note on reproducibility

The train/validation/test partition was generated with a fixed random seed (42), but due to repeated unplanned disconnections in the free-tier Google Colab environment used for training, the underlying dataset directory had to be regenerated more than once during the study. This means the exact image-level composition of the partition was not guaranteed identical across every session, even though the per-class proportions remained constant. The results above reflect a single session in which all four models were trained and evaluated against the same partition, preserving fair comparison between architectures. This is disclosed transparently as a limitation of working in a free, session-based cloud environment rather than persistent infrastructure — see Chapter 3 (§3.3, §3.7) of the thesis for full discussion.

## Repository Structure

leafnet/

├── notebook/

│   └── LeafNet_Training.ipynb       # Full training, evaluation, and benchmarking pipeline

├── app/

│   ├── app.py                       # Flask backend

│   ├── requirements.txt

│   ├── templates/index.html

│   ├── static/style.css

│   └── model/                       # MobileNetV2 weights (see Models section if absent)

├── results/

│   ├── confusion_matrices_all.png

│   ├── inference_speed.txt

│   └── *_results.txt                # Per-model classification reports

└── README.md

## Models

Trained model weights for VGG16 and ResNet50 are not included in this repository due to file size. The MobileNetV2 weights used in the deployed app are included under `app/model/` where size permits; if not present, all four trained models are available at: **[insert your Google Drive link here]**

## Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/leafnet.git
cd leafnet/app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Visit `http://localhost:5000` and upload a potato leaf image.

## Tech Stack

Python, TensorFlow/Keras, scikit-learn, OpenCV, Flask, Pillow — all free and open-source. Trained on Google Colab's free-tier GPU; no paid infrastructure was used at any stage, consistent with the project's goal of a genuinely zero-cost diagnostic tool.

## Citation

If referencing the dataset used in this project, please cite:

Architecture references: He et al. (2016) for ResNet50, Simonyan & Zisserman (2015) for VGG16, Sandler et al. (2018) for MobileNetV2.

## Author

Rajan Koirala — IT Project Thesis, MIT Programme, Himalayan College of Management (Affiliated with IUKL)

## License

This repository's code is provided as-is for academic and research purposes. The PlantVillage dataset is used under its CC BY 4.0 licence; see Mohanty et al. (2016) above for attribution.