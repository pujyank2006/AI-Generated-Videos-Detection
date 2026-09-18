# AI-Generated Video Detection

This repository studies binary classification of real and AI-generated videos using visual and frequency-domain evidence. The project compares RGB-only, FFT-only, blended RGB/FFT, and two-branch hybrid models built around ResNet18.

The project brief is stored at [Report/miniPro.pdf](Report/miniPro.pdf). Its central conclusion is that combining spatial and spectral information is a promising direction for more robust synthetic-video detection.

## Repository

```text
.
├── Models/
│   ├── RGB_model.ipynb              # RGB baseline and video-level voting
│   ├── plainFft_Blend_models.ipynb  # Plain FFT and RGB/FFT blend models
│   └── Hybrid_model.ipynb            # Two-branch RGB + FFT fusion model
├── Report/
│   └── miniPro.pdf                   # Project report and conclusions
└── README.md
```

## Implemented Models

### RGB baseline

`RGB_model.ipynb` fine-tunes a pretrained ImageNet ResNet18 for RGB frame classification. It freezes the earlier backbone layers, trains the final residual block and classifier, and applies image augmentation including flips, rotations, color jitter, blur, random crops, and random erasing.

### FFT models

`plainFft_Blend_models.ipynb` contains two FFT-based experiments using a pretrained ResNet18 with a dropout and multilayer classification head:

- **Plain FFT**: a log-scaled, centered 2-D FFT representation using grayscale magnitude and phase-derived channels.
- **RGB/FFT blend**: combines normalized RGB channels with normalized FFT magnitude using `alpha = 0.5`.

The notebook also includes JPEG compression augmentation, cosine-annealed learning rates, label smoothing, gradient clipping, validation-based checkpointing, and early stopping.

### Hybrid fusion

`Hybrid_model.ipynb` implements a two-stream `HybridResNet`:

1. One pretrained ResNet18 extracts RGB features.
2. A second pretrained ResNet18 extracts FFT features.
3. The two 512-dimensional feature vectors are concatenated.
4. A `512`-unit classifier with ReLU and dropout predicts the two classes.

The hybrid training loop runs for up to 10 epochs with patience 3 and selects the best validation-accuracy checkpoint.

## Processing Pipeline

1. Extract frames from source videos.
2. Divide videos into `train`, `val`, and `test` splits.
3. Keep every frame from one source video in the same split.
4. Load RGB frames or generate FFT representations.
5. Resize inputs to `224 x 224` and apply ImageNet normalization.
6. Train the selected ResNet18-based classifier.
7. Evaluate frame-level predictions with accuracy, precision, recall, F1, classification reports, and confusion matrices.
8. Aggregate frame predictions into video-level predictions when using the voting pipeline.

## Dataset Layout

The dataset and model checkpoints are not included in this repository. The notebooks use Google Drive and Colab paths, with the following general layout:

```text
extracted_frames/
└── frames/
    ├── train/
    │   ├── ai/ or ai_videos/
    │   └── real/ or real_videos/
    ├── val/
    │   ├── ai/ or ai_videos/
    │   └── real/ or real_videos/
    └── test/
        ├── ai/ or ai_videos/
        └── real/ or real_videos/
```

Frames can be nested below each class directory, for example `train/ai/<video_id>/*.jpg`. The exact class names must match the notebook being run: the RGB notebook expects `ai` and `real`, while the standalone FFT/blend notebook uses `ai_videos` and `real_videos`.

The recorded experiments use approximately 25 extracted frames per test video. The FFT/blend and hybrid experiments use 3,498 training frames, 750 validation frames, and 749 test frames.

## Requirements

Install the main dependencies in a Python environment with PyTorch support:

```text
torch
torchvision
numpy
opencv-python
Pillow
matplotlib
seaborn
scikit-learn
tqdm
```

A CUDA-capable GPU is recommended. Every notebook falls back to CPU when CUDA is unavailable. The notebooks were authored for Google Colab and contain Google Drive mounting and shell-copy cells, so local execution may require replacing those cells with local paths.

## Running the Notebooks

1. Upload the extracted dataset to Google Drive or provide an equivalent local path.
2. Open one of the notebooks in Google Colab or Jupyter.
3. Run cells from top to bottom.
4. Update the dataset and output variables when using a different location.
5. Inspect the generated checkpoints, reports, plots, and confusion matrices.

The common configuration is:

| Setting | Value |
| --- | ---: |
| Input size | `224 x 224` |
| Batch size | `32` |
| Epoch limit | `10` |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Random seed | `42` |
| Early-stopping patience | `3` |

The RGB baseline selects checkpoints by validation F1. The FFT/blend and hybrid notebooks select checkpoints by validation accuracy.

## Recorded Results

These values are the outputs saved in the current notebooks, not a guarantee for a new run. Results can change with data, preprocessing, library versions, and hardware.

| Experiment | Evaluation | Accuracy |
| --- | --- | ---: |
| RGB baseline | Test frames, 900 frames | `83.00%` |
| RGB baseline | Test videos, 30 videos | `86.67%` |
| Plain FFT | Test frames, 749 frames | approximately `69%` |
| RGB/FFT blend | Video-level evaluation | `90.00%` |
| Hybrid RGB + FFT | Test frames, 749 frames | `88.79%` |

The RGB baseline also reports best validation F1 of `0.8533`. The hybrid test report gives macro F1 of `0.8875`, with class-specific F1 of `0.8939` for real videos and `0.8810` for AI-generated videos.

## Outputs

Depending on the notebook, training produces:

- PyTorch checkpoints (`.pth`)
- training-history plots
- frame-level classification reports
- video-level voting summaries
- frame-level or video-level confusion matrices

Typical output names include `resnet18_rgb_best.pth`, `ckpt_plain_fft.pth`, `ckpt_fft_rgb.pth`, and `best_hybrid_model_v2.pth`. Output directories are configured inside each notebook and are not committed here.

## Limitations and Future Work

The primary experiments operate on individual frames. They do not fully model motion, temporal consistency, or video-level generation artifacts. The project brief recommends:

- adding LSTM, RNN, or 3D CNN temporal models;
- testing larger and more diverse datasets;
- improving RGB/FFT fusion strategies;
- optimizing inference for real-time or resource-constrained deployment.

## Reference

The report identifies the motivating research direction as **“Beyond Deepfake Images: Detecting AI Generated Videos.”** See [Report/miniPro.pdf](Report/miniPro.pdf) for the complete methodology, comparison, conclusion, and future-work discussion.