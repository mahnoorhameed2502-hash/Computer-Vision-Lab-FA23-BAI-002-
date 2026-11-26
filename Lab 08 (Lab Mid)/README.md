# HOG vs SIFT on Blood-Cell Images

**Computer Vision Lab Mid Exam**
**Name:** Mahnoor Hameed Awan | **Reg No:** FA23-BAI-002 | COMSATS University Islamabad, Wah Campus

This project builds two classical blood-cell classifiers, **HOG + SVM** and **SIFT + SVM**, that run on a normal laptop. It then tests which one stays more reliable when the images are rotated or noisy.

## Dataset

- **BloodMNIST** (`pip install medmnist`), image size 64
- 4 classes: eosinophil, monocyte, neutrophil, platelet
- 100 train, 30 validation and 30 test images per class, chosen with my seed
- Preparation: grayscale, resized to 128 x 128

| Split | Images | Per class |
|---|---|---|
| Train | 400 | 100 |
| Validation | 120 | 30 |
| Test | 120 | 30 |

## My personal settings

These come from my registration number FA23-BAI-002 (last digit 2, last two digits 02).

| Setting | Value |
|---|---|
| Random seed | 2 |
| HOG cell size | 8 x 8 (last digit is even) |
| SIFT grid | 4 x 4 = 16 keypoints (last digit is between 0 and 4) |

## Methods

**HOG + SVM**
- 9 orientations, 8 x 8 cells, 2 x 2 cells per block
- Feature length: 16 x 16 cells, 15 x 15 = 225 blocks, 225 x 36 = **8100**

**SIFT + SVM**
- Normal SIFT gives a different number of keypoints per image (average 70.10, minimum 9, maximum 179, no image with zero), which an SVM cannot use directly.
- Fix: a fixed 4 x 4 grid of keypoints with keypoint size 32, giving one descriptor of 128 numbers per point.
- Feature length: 16 x 128 = **2048**

**Classifier:** `LinearSVC`, with C chosen from {0.1, 1, 10} on the validation set only. A `StandardScaler` is used before the SVM for SIFT because its descriptor values are large.

## Results

**Validation set**

| Model | Best C | Accuracy | Macro-F1 |
|---|---|---|---|
| HOG + SVM | 0.1 | 0.7833 | 0.7837 |
| SIFT + SVM | 1 | 0.7917 | 0.7889 |

**Robustness on the test set** (models frozen; rotation 20 degrees; Gaussian noise sigma = 0.05)

| Model | Clean accuracy | Rotated accuracy | Noisy accuracy |
|---|---|---|---|
| HOG + SVM | 0.7000 | 0.6583 | 0.6083 |
| SIFT + SVM | 0.7083 | 0.7417 | 0.6833 |

## Key findings

- Both models reach similar accuracy on clean images (about 70% on test, about 78 to 79% on validation).
- Platelets are classified perfectly by both models.
- The most confused pair is **eosinophil and monocyte** (13 mistakes for HOG, 11 for SIFT on validation). Eosinophils are mostly told apart by colour, and grayscale removes it.
- **Noise:** HOG dropped 0.0917 and SIFT only 0.0250, so SIFT is clearly more robust.
- **Rotation:** HOG dropped 0.0417, while SIFT went up by 0.0333. The rise is most likely chance, because 120 test images mean a few lucky predictions change accuracy a lot.
- Overall, SIFT with a fixed grid was the more reliable pipeline in this experiment.

## How to run

1. Open `FA23_BAI_002_Mahnoor_Hameed_LabMid_Final.ipynb` in Google Colab.
2. Run all cells (Runtime > Run all). The first cell installs the packages:
   ```
   pip install medmnist scikit-image opencv-python
   ```
3. The dataset downloads automatically (about 156 MB on first run).

## Files

| File | What it is |
|---|---|
| `FA23_BAI_002_Mahnoor_Hameed_LabMid_Final.ipynb` | Executed notebook with all outputs (Q1 to Q5) |
| `FA23_BAI_002_Mahnoor_Hameed_LabMid_Final.md` | Markdown copy of the notebook with outputs |
| `FA23_BAI_002_Mahnoor_Hameed_LabMid_Final.pdf` | PDF copy of the notebook |
| `README.md` | This file |

## Libraries

Python, NumPy, OpenCV, scikit-image, scikit-learn, Matplotlib, pandas, medmnist.

## Notes

- The test set is used only in Q4, after C was chosen on the validation set.
- AI tools were used for help, and the AI log in the notebook (Q5) records what was changed and how it was checked.
