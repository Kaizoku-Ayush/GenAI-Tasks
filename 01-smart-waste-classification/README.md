# 01 · Smart Waste Classification using a Deep Neural Network

An image classifier that sorts waste into **Glass, Paper, Plastic and Metal** to support proper waste segregation and recycling. It is trained on a small, self-collected dataset of mobile-phone photos using **transfer learning** with MobileNetV2.

📓 Notebook: [`smart_waste_classification.ipynb`](smart_waste_classification.ipynb)

## Pipeline

```
image.zip ──► Clean (EXIF fix, RGB, resize 224×224, MD5 de-duplication)
          ──► Stratified split 70 / 15 / 15
          ──► Augment training set only (flip, ±15° rotation, brightness, contrast)
          ──► MobileNetV2 (frozen) + Dense head ──► Train with callbacks
          ──► Evaluate (accuracy, classification report, confusion matrix)
```

## Dataset

| | Glass | Paper | Plastic | Metal | Total |
|---|---|---|---|---|---|
| Original photos | 20 | 20 | 20 | 20 | 80 |
| Train / Val / Test | 14 / 3 / 3 | 14 / 3 / 3 | 14 / 3 / 3 | 14 / 3 / 3 | 56 / 12 / 12 |
| Train after augmentation | 70 | 70 | 70 | 70 | 280 |

Photos were taken from different angles, under different lighting and on different backgrounds. Each class is stored in its own folder.

## Model Architecture

| Layer | Output shape | Params |
|---|---|---|
| Input | 224 × 224 × 3 | 0 |
| Rescaling (pixels → [-1, 1]) | 224 × 224 × 3 | 0 |
| MobileNetV2 (ImageNet, **frozen**) | 7 × 7 × 1280 | 2,257,984 |
| GlobalAveragePooling2D | 1280 | 0 |
| Dropout (0.3) | 1280 | 0 |
| Dense (64, ReLU) | 64 | 81,984 |
| Dropout (0.2) | 64 | 0 |
| Dense (4, Softmax) | 4 | 260 |

**Total:** 2,340,228 parameters, of which **82,244 are trainable**.

## Training

- **Optimizer:** Adam (learning rate 0.0005)
- **Loss:** Sparse Categorical Cross-Entropy
- **Batch size / epochs:** 16 / up to 20
- **Callbacks:**
  - `ModelCheckpoint` (best `val_loss`)
  - `EarlyStopping` (patience 5, restore best weights)
  - `ReduceLROnPlateau` (factor 0.3, patience 2)
- Training stopped early at epoch 10. The best weights came from epoch 5.

## Results

| Split | Loss | Accuracy |
|---|---|---|
| Validation | 0.5558 | 83.33% |
| Test | 0.0953 | **100.00% (12/12)** |

Precision, recall and F1 are 1.00 for every class on the test set, and the confusion matrix is perfectly diagonal.

> The test set is small (3 images per class), so a larger and more diverse dataset would give a more reliable estimate.

## Requirements

`tensorflow`, `pillow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn` (all available in Google Colab by default).

## How to Run

1. Open the notebook in Google Colab.
2. Run the first cell and upload `image.zip`, which contains the folders `Glass/`, `Paper/`, `Plastic/` and `Metal/`.
3. Run the remaining cells in order.
