# CNN Feature Visualization, Parameter Audit & Critical Analysis

A practical lab on how convolutional neural networks (CNNs) transform an image, step by step, and how their parameter counts follow from their architecture. The notebook builds the operations by hand in NumPy, visualizes each stage, checks the parameter counts against a Keras model, and works through a set of conceptual questions on activation functions, pooling, and parameter efficiency.

**Course:** Deep Learning & Computer Vision
**Student:** Mahnoor Hameed
**Submission date:** 26 November 2026

## Contents

The notebook has three parts.

**Part I: Filter pipeline and visualization.** Five 3×3 kernels (horizontal edge, vertical edge, sharpen, box blur, emboss) are applied to a grayscale image. Each kernel goes through four stages:

- **A: Raw convolution.** Cross-correlation with zero padding and stride 1, so the output keeps the input size.
- **B: ReLU.** Negative responses are set to zero.
- **C: Max pooling.** 2×2 windows with stride 2.
- **D: Average pooling.** 2×2 windows with stride 2.

The results are shown in a 5×4 grid, and a table reports the share of negative and zeroed activations, the min and max of the raw responses, and the mean and peak values after each pooling type.

**Part II: Architecture comparison and parameter audit.** Two small CNNs are compared. Both take a 32×32×3 input and follow the same layout: Conv1 (5 filters), pooling, Conv2 (10 filters), pooling, global average pooling, and a Dense(10) softmax output.

- **Model A:** 3×3 kernels with max pooling.
- **Model B:** 5×5 kernels with average pooling.

Parameter counts are calculated by hand and then checked against `model.count_params()` in Keras, with an assertion that the two agree.

**Part III: Critical thinking.** Four questions are explored with short experiments:

1. **ReLU "blackout."** Compares ReLU, Leaky ReLU (α = 0.1), and sigmoid on the same feature map.
2. **Max vs. average pooling.** Shows how a one-pixel-wide dark stroke is erased by max pooling but preserved as gray by average pooling.
3. **Parameter explosion and receptive fields.** Measures the parameter increase from Model A to Model B and the savings from stacking two 3×3 convolutions in place of one 5×5.
4. **Order invariance.** Checks whether ReLU and max pooling commute (they do for max pooling, but not for average pooling).

## Key results

| Quantity | Value |
|---|---|
| Model A total parameters (3×3, max pool) | 710 |
| Model B total parameters (5×5, avg pool) | 1,750 |
| Increase from A to B | 146.48% |
| Parameters saved by stacking two 3×3 convs instead of one 5×5 (both layers) | 726 (41.5% of Model B) |
| Sigmoid output range on the Sobel feature map | 0.033 to 0.958 |
| Average-pooled value of a 1-pixel stroke in a 2×2 window | 63.75 (max pooling gives 255, i.e. erased) |

Manual and code-based counts match for both models.

## Requirements

- Python 3
- NumPy
- Matplotlib
- SciPy (`scipy.signal.correlate2d`)
- scikit-image (`skimage.data`, `color`, `io`, `transform`)
- pandas
- TensorFlow / Keras

The notebook was written for Google Colab, where all of these are preinstalled.

## How to run

1. Open the notebook in Google Colab or Jupyter.
2. Run the cells in order from top to bottom. Part II depends on the NumPy helper functions defined in Part I, and Part III reuses the feature maps computed earlier.
3. To use your own image, set `USE_UPLOAD = True` in the input cell. In Colab, you will be prompted to upload a file. Any RGB or grayscale image works, and it is converted to grayscale and resized to 256×256.

By default the notebook uses the built-in `skimage` camera image, so it runs without any uploads.

## Notes

- The first Sobel kernel is labelled **Horizontal Edge** because it responds to intensity changes along the vertical direction. The handout calls it "Sobel-X/Prewitt." In the textbook convention, these two kernels are Sobel-Y and Sobel-X respectively. The labels were chosen to match behaviour.
- Convolution is implemented as cross-correlation, which is what CNN libraries compute. The convention matters for sign but not for the parameter counts.
- The pooling function is written as a Python loop for clarity, so it is slow on large inputs. For 256×256 images it runs quickly enough.
