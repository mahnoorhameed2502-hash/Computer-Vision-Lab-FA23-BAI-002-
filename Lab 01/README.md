# Skin Lesion Classification: Transfer Learning Comparison

This notebook compares transfer-learning backbones and classical classifiers for skin lesion classification on the **Skin Cancer ISIC (9 classes)** dataset. It produces three tables:

- **Table 1:** eight ImageNet-pretrained CNNs, each fine-tuned end-to-end as a classifier
- **Table 2:** classical classifiers trained on deep features extracted from the fine-tuned ResNet50
- **Table 3:** computational cost of each backbone (parameters, model size, FLOPs, inference time) alongside its Table 1 accuracy

**Notebook:** `Lab_01_CV_Skin_Lesion_Classification_Colab__1___1_.ipynb`

**Dataset:** https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic

## Pipeline

1. **Install dependencies:** kagglehub, thop, xgboost, scikit-learn.
2. **Download the dataset** with kagglehub. The download contains `Train/` and `Test/` folders, each with one sub-folder per class.
3. **Build a file list and re-split.** All images from both folders are pooled and split 80 / 10 / 10 with stratification by class and `random_state=42`. The original `Test/` folder is not kept as the test set.
4. **Create datasets and loaders:** images are resized to 224×224, normalised with ImageNet statistics, and batched at 32. Training images receive augmentation: horizontal and vertical flips, random affine translation, shear, and scaling.
5. **Train and evaluate the eight backbones (Table 1).**
6. **Extract features and train seven classifiers (Table 2).**
7. **Profile efficiency (Table 3).**
8. **Save all tables** as CSV files and download them (Colab only).

## Data split

The 2,357 images across nine classes are split as follows:

| Split | Images |
|---|---|
| Train | 1,885 |
| Validation | 236 |
| Test | 236 |

Class sizes range from 80 images (seborrheic keratosis) to 478 (pigmented benign keratosis). The split is stratified, so each split keeps the same class proportions.

## Models

### Table 1: Fine-tuned backbones

Eight architectures are loaded with ImageNet weights, and the final classification layer is replaced for nine classes:

AlexNet, VGG16, VGG19, ResNet18, ResNet50, ResNet101, DenseNet121, EfficientNet-B0.

Training uses cross-entropy loss and the Adam optimiser with a learning rate of 1e-4:

1. **Phase 1 (2 epochs):** the backbone is frozen and only the final layer is trained.
2. **Phase 2 (10 epochs by default):** all layers are unfrozen and trained at a learning rate of 1e-5. A plateau scheduler reduces the rate when validation loss stops improving. The checkpoint with the lowest validation loss is kept.

Test metrics are weighted precision, recall, and F1, along with accuracy and one-vs-rest AUC.

### Table 2: Classical classifiers on deep features

The fine-tuned ResNet50 from Table 1 is used as a fixed feature extractor. Its final layer is removed, leaving 2,048-dimensional pooled features. Validation features are merged into the training set, which gives 2,121 training samples.

Features are standardised, and seven classifiers are trained: Logistic Regression, Decision Tree, Random Forest (300 trees), KNN (k=5), Linear SVM, RBF-SVM, and XGBoost (300 estimators). Metrics are weighted across classes.

### Table 3: Efficiency

For each backbone, the notebook reports:

- parameter count (millions)
- saved model size (MB)
- FLOPs for one 224×224 forward pass (computed as 2 × MACs from `thop`)
- average inference time per image at batch size 1, after 5 warm-up runs and averaged over 30 runs
- Table 1 test accuracy

## Results from the default run (10 epochs)

### Table 1: Transfer-learning backbones (test set, %)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| AlexNet | 66.53 | 66.06 | 66.53 | 64.86 | 92.68 |
| VGG16 | 66.95 | 66.30 | 66.95 | 65.26 | 92.52 |
| VGG19 | 67.37 | 67.62 | 67.37 | 66.45 | 93.22 |
| ResNet18 | 72.88 | 70.61 | 72.88 | 71.05 | 94.35 |
| ResNet50 | 74.58 | 73.34 | 74.58 | 73.40 | 95.23 |
| **ResNet101** | **76.69** | **74.74** | **76.69** | **75.45** | **95.79** |
| DenseNet121 | 71.19 | 70.53 | 71.19 | 69.85 | 93.92 |
| EfficientNet-B0 | 58.90 | 57.87 | 58.90 | 53.93 | 89.34 |

### Table 2: Classifiers on ResNet50 deep features (test set, %)

| Classifier | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 69.07 | 69.14 | 69.07 | 68.21 | 93.47 |
| Decision Tree | 57.20 | 57.90 | 57.20 | 57.31 | 75.03 |
| Random Forest | 75.42 | 74.10 | 75.42 | 74.37 | 94.59 |
| KNN (k=5) | 74.15 | 76.08 | 74.15 | 74.33 | 91.15 |
| Linear SVM | 72.03 | 71.61 | 72.03 | 71.41 | 94.61 |
| **RBF-SVM** | **76.27** | 74.68 | **76.27** | **75.09** | 94.65 |
| XGBoost | 75.85 | 74.46 | 75.85 | 74.86 | 94.00 |

### Table 3: Computational efficiency

| Model | Params (M) | Size (MB) | FLOPs (G) | Inference (ms) | Accuracy (%) |
|---|---|---|---|---|---|
| AlexNet | 57.04 | 217.60 | 1.42 | 2.11 | 66.53 |
| VGG16 | 134.30 | 512.32 | 30.93 | 11.32 | 66.95 |
| VGG19 | 139.61 | 532.57 | 39.26 | 13.58 | 67.37 |
| ResNet18 | 11.18 | 42.73 | 3.65 | 3.80 | 72.88 |
| ResNet50 | 23.53 | 90.06 | 8.26 | 5.99 | 74.58 |
| ResNet101 | 42.52 | 162.82 | 15.73 | 17.46 | 76.69 |
| DenseNet121 | 6.96 | 27.18 | 5.79 | 21.88 | 71.19 |
| EfficientNet-B0 | 4.02 | 15.71 | 0.83 | 7.89 | 58.90 |

In this run, ResNet101 gave the highest Table 1 accuracy, and RBF-SVM on ResNet50 features gave the highest Table 2 accuracy. EfficientNet-B0 had the fewest parameters and FLOPs but the lowest accuracy. ResNet18 and ResNet50 offer a middle ground between accuracy and cost.

**Caution on interpretation:** the test set has only 236 images, and the rarest classes have very few examples. A difference of one or two percentage points is about two to five images, so small gaps between models may not be meaningful. The results also come from a single run with one random seed and only 10 fine-tuning epochs.

## Requirements

- Python 3.9 or newer (the notebook was run on Python 3.13)
- PyTorch and torchvision, with CUDA recommended
- kagglehub, thop, xgboost, scikit-learn, pandas, numpy, pillow
- A Kaggle setup that lets kagglehub download the dataset (Colab usually handles this)

The notebook was written for Google Colab. It uses `google.colab.files` to download the CSVs in the final cell. Outside Colab, skip that cell and read the CSVs from disk.

## How to run

1. Open the notebook in Google Colab, with a GPU runtime selected for reasonable training times.
2. Run **Runtime → Run all**. The first run downloads the dataset and the ImageNet weights for each backbone.
3. To train longer, change `EPOCHS` in the Table 1 cell. The notebook suggests 20 to 30 for a final result. Longer training will take proportionally more time.

Training all eight backbones at 10 epochs takes a while on a Colab GPU. The notebook prints per-epoch losses so you can follow progress.

## Outputs

| File | Contents |
|---|---|
| `table1_transfer_learning_models.csv` | Table 1 |
| `table2_classifier_comparison.csv` | Table 2 |
| `table3_computational_efficiency.csv` | Table 3 |

## Notes on method

- The validation set is used for checkpoint selection in Table 1 and is merged into the training data for Table 2. The test set is used only for reported metrics.
- Table 1 uses weighted averages across classes. Precision, recall, and F1 are therefore dominated by the larger classes.
- Inference times depend on the hardware. The numbers above were measured on a CUDA GPU, so they will differ on CPU or another GPU.
- The Table 2 features come from a ResNet50 fine-tuned on the training split only, so the classical classifiers see features that were learned without the test data.
