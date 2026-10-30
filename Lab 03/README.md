# Lab 03: Edge Detection Techniques and Their Impact on Classification Performance

This lab studies how classical edge detectors change the performance of image classifiers. It compares five edge operators, measures how they behave under noise, tunes the Canny detector, and then trains classifiers on three versions of the same dataset:

- **Set A (Raw):** original grayscale images
- **Set B (Filtered):** images after the best Lab 02 filter (Gaussian, median, or bilateral)
- **Set C (Edge):** the best edge representation, selected automatically

Each classifier is trained and evaluated on all three sets using the same split, epochs, and metrics.

**Notebook:** `Lab_03_Edge_Detection_Classification.ipynb`

## Pipeline

1. **Setup:** load libraries and set the configuration in `CFG`.
2. **Dataset:** load grayscale images, resize them to 64×64, and create a fixed train/validation/test split (60/20/20, stratified).
3. **Task 1, Comparative edge detection:** Sobel, Prewitt, Laplacian, Laplacian of Gaussian (LoG), and Canny applied to one representative image per class. The notebook also shows the first-order components (Sobel Gx, Gy, and magnitude).
4. **Task 2, Noise study (Table 1):** each detector is tested on clean images, images with Gaussian noise (σ=25), images with salt-and-pepper noise (5%), and the noisy images after Gaussian or median filtering. Results are measured against the clean edge map using false edges, broken edges, edge F1, edge-density ratio, and segment-count ratio.
5. **Task 3, Canny parameter study (Table 2):** six combinations of low/high thresholds and kernel size are compared by validation macro-F1 of a linear classifier on 16×16 edge maps, along with edge density and fragmentation.
6. **Choosing Set C:** all five edge detectors are scored with the same quick test. The highest-scoring one becomes Set C.
7. **Task 4, Dataset construction:** build Sets A, B, and C and display examples.
8. **Task 5, Training and evaluation (Table 3):** train SVM, Random Forest, KNN, and two CNNs on each set. Hyperparameters and the best epoch are chosen on validation data. Test accuracy, macro precision, recall, F1, training time, and inference time are reported.
9. **Task 6, Visual comparison:** confusion matrices and a bar chart for the best model, chosen by mean validation accuracy.
10. **Summary and discussion:** an auto-generated summary of whether edge detection helped, followed by written analysis.

## Models

| Model | Input | Notes |
|---|---|---|
| SVM (RBF) | 32×32 pixels, standardised | `C` chosen from {1, 10} on validation |
| Random Forest | 32×32 pixels | 200 trees |
| KNN | 32×32 pixels, standardised | `k` chosen from {3, 5, 7} on validation |
| CNN Model 1 | 64×64 image | 2 conv blocks, average pooling, 2 linear layers |
| CNN Model 2 | 64×64 image | 3 conv blocks with batch norm, global pooling, dropout 0.3 |

Both CNNs use Adam (lr 1e-3), batch size 64, and 15 epochs, and keep the checkpoint with the best validation accuracy.

## Results from the default run (CIFAR-10, 4 classes)

Default classes: airplane, automobile, cat, ship. 500 images per class, split 1200 / 400 / 400.

- **Best edge representation:** Canny, 50/150 thresholds, 3×3 blur kernel (Canny-2).
- **Best model:** CNN Model 2, selected by mean validation accuracy.

| Dataset | CNN Model 2 test accuracy |
|---|---|
| Raw (Set A) | 69.00% |
| Filtered (Set B, Gaussian) | 65.25% |
| Edge (Set C, Canny) | 64.50% |

On this data, edge detection reduced accuracy by about 4.5 percentage points for the best model. Results on your own dataset may differ. The notebook prints the exact conclusion for your run in its auto-generated summary.

## Requirements

- Python 3.9 or newer
- numpy, pandas, matplotlib, scipy, scikit-learn, scikit-image
- opencv-python-headless (installed by the first cell)
- torch, torchvision (torchvision is needed only for the CIFAR-10 demo)

The notebook was written for Google Colab, which already includes most of these packages. A GPU is optional. The code selects CUDA when it is available and falls back to CPU otherwise.

## How to run

1. Open the notebook in Google Colab or Jupyter.
2. Run **Runtime → Run all**. The default CIFAR-10 mode downloads about 170 MB on first run.
3. To use your own dataset (for example, the Lab 01/02 data), edit the configuration cell:

   ```python
   CFG = dict(
       data_mode="folder",              # use your own folder of images
       data_dir="/content/dataset",     # one sub-folder per class
       classes=None,                    # None = use all sub-folders
       best_filter="gaussian",          # your Lab 02 winner
       ...
   )
   ```

   Each class needs its own sub-folder containing `.png`, `.jpg`, `.jpeg`, `.bmp`, or `.tif` files.

Other options for `data_mode` are `"cifar10"` (default demo) and `"digits"` (a small offline scikit-learn dataset for testing).

## Outputs

All outputs are written to a `results/` folder:

| File | Contents |
|---|---|
| `table1_noise.csv` | Table 1: edge detector robustness to noise |
| `noise_full_matrix.csv` | Full per-detector, per-condition metrics |
| `table2_canny.csv` | Table 2: Canny parameter study |
| `edge_method_comparison.csv` | Validation scores for each edge representation |
| `table3_cross_lab.csv` | Table 3: accuracy on Raw, Filtered, and Edge sets |
| `table3b_full_results.csv` | Full results for every model and dataset |
| `task1_edge_comparison.png`, `task1_first_order.png` | Edge detector visual comparison |
| `task2_noise_visual.png`, `task2_robustness.png` | Noise examples and robustness chart |
| `task3_canny_params.png` | Canny parameter examples |
| `task4_datasets.png` | Examples of Sets A, B, and C |
| `task6_confusion_matrices.png`, `task6_metric_bars.png` | Confusion matrices and metric comparison |
| `cnn_vs_edge_kernels.png` | Learned CNN filters next to Sobel and Prewitt kernels |

## Reproducibility

The random seed is fixed at 42 for Python, NumPy, and PyTorch. Data splits, model initialisation, and batch order all use this seed, so repeated runs on the same machine should give the same results. Small differences may still occur across hardware or library versions.

## Notes on method

- All hyperparameter and checkpoint choices use the validation set. The test set is used only for reported numbers.
- Edge maps are binarised with Otsu's threshold on the clean image. The same threshold is applied to every noisy condition.
- Precision and recall for edge maps allow a 1-pixel tolerance through dilation.
- Noise sensitivity is labelled by the relative loss in edge F1: Low (<25%), Medium (<50%), or High (≥50%).
