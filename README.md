# AI-Generated Video Detection

This project investigates whether spatial and frequency-domain features can distinguish real videos from AI-generated videos. It compares three approaches described in `miniPro.pdf`:

- **RGB**: learns visible spatial artifacts directly from video frames.
- **FFT**: learns frequency-domain artifacts that may be introduced during generation.
- **Fusion (RGB + FFT)**: combines complementary spatial and spectral information.

The project brief reports that the fusion approach provides the best overall trade-off between accuracy and robustness, while the current repository contains the RGB and FFT notebook implementations used in the study.

## Repository Layout

```text
.
├── Models/
│   ├── rgb.ipynb    # RGB frame classifier and evaluation pipeline
│   └── fft.ipynb    # FFT representations, FFT classifiers, and video voting
├── miniPro.pdf      # Project brief, comparison, conclusion, and future work
└── README.md
```

## Method

Both notebooks perform frame-level binary classification with a pretrained **ResNet18** backbone:

1. Video files are represented as extracted image frames.
2. Frames are organized into `train`, `val`, and `test` splits.
3. Each split contains `ai` and `real` class directories.
4. Models are trained and evaluated using accuracy, F1 score, classification reports, and confusion matrices.
5. Test predictions can also be aggregated at video level by voting across frames.

The RGB notebook uses ImageNet normalization and data augmentation for natural images. The FFT notebook computes log-scaled 2-D FFT magnitude spectra and evaluates multiple spectral representations, including plain FFT, FFT/RGB combinations, and per-channel FFT features.

## Dataset Structure

The dataset is not included in this repository. Prepare extracted frames in the following structure:

```text
extracted_frames/
└── frames/
    ├── train/
    │   ├── ai/
    │   │   └── <video_id>/*.jpg
    │   └── real/
    │       └── <video_id>/*.jpg
    ├── val/
    │   ├── ai/
    │   └── real/
    └── test/
        ├── ai/
        └── real/
```

The notebooks recursively search the class directories, so frames may be stored inside per-video folders. Keep frames from the same source video in the same split to avoid data leakage.

## Requirements

The notebooks use Python and the following main packages:

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

A CUDA-capable GPU is recommended, but both notebooks fall back to CPU when CUDA is unavailable.

## Running the Notebooks

The notebooks were written for Google Colab and currently use Google Drive paths. To run them:

1. Upload or mount the extracted-frame dataset in Google Drive.
2. Open `Models/rgb.ipynb` or `Models/fft.ipynb` in Colab.
3. Update `DATA_DIR`, `DATASET_ROOT`, `TEST_DIR`, and `OUTPUT_DIR` to match the location of your data.
4. Run the cells from top to bottom.
5. Review the saved checkpoints, classification reports, and confusion matrices in the configured output directory.

The main training settings are `224 × 224` input images, batch size `32`, learning rate `1e-4`, weight decay `1e-4`, seed `42`, and up to `10` epochs. The RGB pipeline selects the best checkpoint by validation F1; the FFT pipeline selects it by validation accuracy.

## Expected Outputs

Training and evaluation produce:

- best model checkpoints (`.pth`)
- frame-level accuracy and classification reports
- video-level accuracy when frame voting is enabled
- frame-level and video-level confusion-matrix images

Results depend on the dataset, split, hardware, and preprocessing configuration. The project brief characterizes RGB as a moderate baseline, FFT as stronger for hidden spectral artifacts, and fusion as the strongest of the three compared approaches.

## Limitations and Future Work

The current approach analyzes individual frames and therefore does not model motion or temporal consistency. The brief identifies the following directions for improvement:

- add LSTM, RNN, or 3D CNN temporal models;
- evaluate larger and more diverse datasets;
- investigate stronger RGB/FFT fusion strategies;
- optimize the models for real-time or resource-constrained deployment.

## Reference

The project brief identifies the motivating research direction as **“Beyond Deepfake Images: Detecting AI Generated Videos.”** See `miniPro.pdf` for the full project description, comparative discussion, conclusion, and reported findings.