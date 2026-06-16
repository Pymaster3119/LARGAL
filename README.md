# LARGAL: Radio Galaxy Detection and Classification Pipeline

**By Aditya Anand**

LARGAL is a machine learning pipeline for detecting and classifying radio galaxies from astronomical images. It combines object detection (YOLOv8s), dimensionality reduction (Variational Autoencoder), and classification (Histogram Gradient Boosting) into a comprehensive end-to-end workflow. This work has been published in Vanderbilt's Young Scientist Journal, available here: https://wp0.vanderbilt.edu/youngscientistjournal/article/largal-learning-latent-representation-of-radio-galaxies-for-efficient-compression-and-improved-classification

## Overview

LARGAL implements a three-stage detection and classification pipeline:

1. **Object Detection (YOLOv8s)**: Locates radio galaxies in images
2. **Feature Extraction (VAE)**: Encodes detected galaxies into a compact latent space
3. **Classification**: Trains classifiers of various types (e.g., Histogram Gradient Boosting, Random Forest, SVM, etc.) on latent representations

The system handles galaxies of different sizes separately (small: area < 24² pixels, medium: area ≥ 24² pixels) for improved classification accuracy.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU for trainable
- pip or conda

### Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd LARGAL
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download VAE weights here: https://drive.google.com/file/d/1yrI4OxyD2WN8Fu1mKQ8B5AZVo3zXMhqj/view?usp=sharing

### Dependencies

- **numpy**: Numerical computing
- **scikit-learn**: Machine learning classifiers and preprocessing
- **xgboost**: Gradient boosting algorithms
- **joblib**: Model serialization
- **torch**: Deep learning framework (for VAE)
- **opencv-python**: Image processing
- **ultralytics**: YOLOv8 implementation

## Project Structure

```
LARGAL/
├── README.md 
├── REAMEfigures/                         # Figures used in README.md
│   ├── t-SNE Plot.jpg                    # t-SNE plot of latent vectors
│   └── VAE Reconstructions.jpg           # Sample variational autoencoder reconstructions, dataset images on top and reconstructions blow.
│
├── requirements.txt                       # Python dependencies
├── dataset.yaml                           # YOLOv8 dataset configuration
├── dataset/                               # Training data
│   ├── images/
│   │   ├── train/                        # Training images
│   │   ├── val/                          # Validation images
│   │   └── test/                         # Test images
│   └── labels/
│       ├── train/                        # YOLO format annotations
│       ├── val/
│       └── test/
│
├── yolov8s-finetune/                     # Fine-tuned YOLO model
│   ├── weights/
│   │   ├── best.pt                       # Best model checkpoint
│   │   └── last.pt                       # Last checkpoint
│   ├── args.yaml                         # Training arguments
│   └── results/                          # Training metrics and crops
│   ... Other result files not listed
│
├── Notebooks (Jupyter):
│   ├── finetune_yolo.ipynb               # Script to fine-tune YOLOv8s model
│   ├── trainVAE.ipynb                    # Script to train VAE model
│   ├── clusterizeVAE.ipynb               # Create VAE clusters (auxilary experiment not used in final result)
│   └── ConvertDatasetToObjects.ipynb     # Dataset preprocessing (auxilary script not used in final result)
│
├── Scripts (Python):
│   ├── docrops.py                        # Crop radio galaxies based on ground truth
│   ├── extract_latents.py                # Extract VAE latent vectors for classifier training
│   ├── train_classifiers_multisize.py    # Train all classification models
│   ├── evaluate_bboxes_multisize.py      # Evaluate full pipeline (mAP)
│   ├── run_tsne_latents_valcubic.py      # Visualize latents with t-SNE
│   ├── dataparsing.py                    # Data parsing utilities
│   ├── loss.py                           # VAE loss functions
│   ├── compute_map.py                    # mAP computation helper
│   └── evaluate_bboxes.py                # Single-size evaluation
│
└── Output directories (created during execution):
    ├── latents/                          # Extracted latent vectors
    ├── classifiers/                      # Trained classifier models
    ├── VAEModelCubicFit.pth              # Trained VAE weights
    └── tsne_plots/                       # Dimensionality reduction plots
```

## Training Pipeline

### Step 1: Prepare Dataset

Update `dataset.yaml` with your dataset paths.
Follow this template for default values:
```yaml
path: /path/to/dataset
train: dataset/images/train
val: dataset/images/val
test: dataset/images/test
nc: 1
names:
  - Radio Galaxy
```

### Step 2: Fine-tune YOLO Object Detector

Run the `finetune_yolo.ipynb` notebook to fine-tune YOLOv8s on your radio galaxy images:
- Detects radio galaxies and produces bounding boxes
- Outputs are saved in `yolov8s-finetune/weights/best.pt`
- Results stored in `yolov8s-finetune/results/`

### Step 3: Crop Detected Galaxies

Execute `docrops.py` to extract galaxy crops:
```bash
python docrops.py
```
- Creates three pickle files: `traincubicfit.pkl`, `testcubicfit.pkl`, `valcubicfit.pkl`
- Removes background noise and prepares crops for VAE training

### Step 4: Train VAE

Run `trainVAE.ipynb` or `clusterizeVAE.ipynb`:
- Trains a Variational Autoencoder with self-attention layers
- Compresses galaxy images into a latent vector space
- Outputs model weights to `VAEModelCubicFit.pth`
- Uses custom VAE loss function (MSE reconstruction + KL divergence)

### Step 5: Extract Latent Representations

Execute `extract_latents.py`:
```bash
python extract_latents.py
```
- Passes cropped galaxies through the trained VAE
- Generates latent vectors (compressed representations)
- Creates output files in `latents/`:
  - `latents_train.npy`, `latents_val.npy`, `latents_test.npy`
  - `labels_train.npy`, `labels_val.npy`, `labels_test.npy`
  - `areas_train.npy`, `areas_val.npy`, `areas_test.npy`

### Step 6: Train Size-Aware Classifiers

Execute `train_classifiers_multisize.py`:
```bash
python train_classifiers_multisize.py
```
- Trains separate classifiers for small (area < 24²) and medium (area ≥ 24²) galaxies
- Models supported: Random Forest, Histogram Gradient Boosting, K-Nearest Neighbors, Logistic Regression, Ridge, SVM
- Includes optional SMOTE for class imbalance handling
- Saves trained models to `classifiers/`

### Step 7: Evaluate Full Pipeline

Execute `evaluate_bboxes_multisize.py` to compute mAP:
```bash
python evaluate_bboxes_multisize.py
```
- Evaluates end-to-end detection → feature extraction → classification pipeline
- Produces mAP (mean Average Precision) score
- Generates detailed performance metrics

### Step 8: Visualize Latent Space

Execute `run_tsne_latents_valcubic.py`:
```bash
python run_tsne_latents_valcubic.py
```
- Creates t-SNE visualization of latent space
- Helps understand how VAE clusters galaxies
- Outputs plots to `tsne_plots/`

## Evaluation Pipeline

The evaluation flow uses the saved VAE weights and classifier models already included in the repository. Run `evaluate_bboxes_multisize.py` to score the full pipeline end to end, and use `run_tsne_latents_valcubic.py` if you want a quick latent-space visualization.

Expected files:
- VAE weights: `VAEModelCubicFit.pth`
- Classifiers: `classifiers/`
- Detection weights: `yolov8s-finetune/weights/best.pt`

## Technical Details

### Architecture Components

**YOLOv8s Object Detector**
- Small variant of YOLO v8 for resource efficiency
- Trained on radio galaxy detection task
- Produces bounding box coordinates and confidence scores

**Variational Autoencoder (VAE)**
- Encoder: Convolutional layers with self-attention
- Latent space: Compressed representation of galaxies
- Decoder: Reconstructs original images from latents
- Loss function: Reconstruction MSE + KL divergence penalty
- Output: Latent vectors typically 50-500 dimensions (configurable)

**Size-Aware Classifiers**
- **Small galaxies** (area < 576 pixels): Optimized for compact detections
- **Medium galaxies** (area ≥ 576 pixels): Separate model for larger galaxies
- **Methods**: Histogram Gradient Boosting (default), Random Forest, SVM, etc.
- **Features**: Latent vectors from VAE encoder

## Results

The VAE reduces the input size by 95.8% while still reconstructing the core structure of each radio galaxy, with a mean squared reconstruction error of 159.35. Its latent space also separates galaxy classes well, as shown by t-SNE projections of the validation set.

The classifier trained on VAE latent vectors achieved 76.43% validation accuracy and 81.38% test accuracy for small objects, while the medium and large object model reached 77.19% validation accuracy and 78.64% test accuracy. Histogram Gradient Boosting performed best across the tested models.

The full detection pipeline achieved 61.7% mAP at IoU 0.5, 60.0% mAP at IoU 0.75, and 53.5% mAP averaged across IoU thresholds from 0.05 to 0.95. Overall, the dataset was compressed from 247.7 MB to 2.12 MB while preserving strong detection and classification performance.

### Visualizations

![Figure 4: t-SNE plot of latent vectors](READMEfigures/t-SNE%20Plot.jpg)

**Figure 4.** t-SNE plot of latent vectors from the validation set.

![Figure 5: VAE reconstructions](READMEfigures/VAE%20Reconstructions.jpg)

**Figure 5.** VAE reconstructions of radio galaxy images. Dataset images are on the top row, while reconstructions are below.
