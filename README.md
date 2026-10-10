# Covariate-Guided Reconstruction of Missing Geochemical Data

Code release for the manuscript:

> **Covariate-guided reconstruction of missing geochemical data: An
> application to mineral prospectivity mapping**
>
> Siyuan Ban, Zhenjie Zhang, Fuyuan Jian, Kexin Zhou
>
> China University of Geosciences (Beijing)
>
> E-mail: zjzhang@cugb.edu.cn
>
> *Computers & Geosciences* (manuscript)

This package implements two variants of a Recurrent Feature Reasoning
Network (RFR-Net) for reconstructing missing regions of regional
geochemical rasters:

| Package     | Covariate inputs | Description |
|-------------|------------------|-------------|
| `RFR`       | No               | Baseline model. Only the target geochemical variable is used as network input. |
| `RFR_all`  | Yes (multiple)   | Covariate-guided model. An arbitrary number of covariate layers (e.g. DEM, aeromagnetic data, fault proximity) is concatenated channel-wise to the input of the first partial-convolution layer. |

The two variants share the same training/testing workflow, losses and
overall architecture; the only difference is the covariate fusion at the
network input (`RFR_all/run.py --n_ext_feat` controls the number of
covariates).

## Contents

```
.
├── README.md                     # This file
├── LICENSE                       # License (see note below)
├── requirements.txt              # Python dependencies (pip)
├── environment.yml               # Conda environment
├── generate_synthetic_data.py    # Synthetic example dataset generator
├── data/                         # Data directory (template, see data/README.md)
├── RFR/                          # Baseline model (no covariates)
│   ├── run.py                    # Command-line entry point
│   ├── model.py                  # Training/testing logic and losses
│   ├── dataset.py                # PyTorch dataset (images + masks)
│   ├── modules/
│   │   ├── RFRNet.py              # Network architecture
│   │   ├── partialconv2d.py       # Partial convolution (NVIDIA, BSD-3)
│   │   └── Attention.py          # Knowledge-consistent attention
│   └── utils/
│       └── io.py                  # Checkpoint save/load helpers
└── RFR_all/                     # Covariate-guided model
    ├── run.py                    # Command-line entry point
    ├── model.py                  # Training/testing logic, losses, metrics
    ├── dataset.py                # PyTorch dataset (images + masks + covariates)
    ├── modules/
    │   ├── RFRNet.py              # Network architecture (covariate fusion)
    │   ├── partialconv2d.py       # Partial convolution (NVIDIA, BSD-3)
    │   └── Attention.py          # Knowledge-consistent attention
    └── utils/
        └── io.py                  # Checkpoint save/load helpers
```

## Requirements

* **Operating system**: Windows / Linux / macOS (tested on Windows 10/11).
* **Python**: 3.8 or newer.
* **Hardware**: an NVIDIA GPU with CUDA support is strongly recommended
  (the code automatically falls back to CPU if CUDA is unavailable, but
  training will be very slow).
* **Python packages** (see `requirements.txt`):

  - `torch`, `torchvision` (with CUDA build if a GPU is used)
  - `numpy`
  - `opencv-python`
  - `rasterio`
  - `matplotlib`
  - `scikit-learn >= 1.0, < 1.4` (the `squared` argument of
    `mean_squared_error` was removed in scikit-learn 1.4)
  - `scipy`
  - `Pillow`
  - `tensorboard`

## Installation

Option 1 — pip:

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

Option 2 — conda (recommended for `rasterio` on Windows):

```bash
conda env create -f environment.yml
conda activate rfr-geochem
```

## Data availability

The real geochemical survey data used in the manuscript are confidential
and **cannot be shared**. To make the workflow reproducible, the script
`generate_synthetic_data.py` creates a synthetic example dataset with
exactly the same directory structure. Users can replace the synthetic
rasters with their own single-band GeoTIFF data following the layout
described in `data/README.md`.

## Usage

### 1. Generate the synthetic example dataset

```bash
python generate_synthetic_data.py --out ./data --n_train 40 --n_test 10 --size 128
```

### 2. Train the baseline model (`RFR`, no covariates)

```bash
cd RFR
python run.py \
    --data_root    ../data/train/images \
    --mask_root    ../data/train/masks \
    --model_save_path ./checkpoints \
    --num_iters   50000
```

### 3. Train the covariate-guided model (`RFR_all`)

```bash
cd RFR_all
python run.py \
    --data_root      ../data/train/images \
    --mask_root      ../data/train/masks \
    --covariate_root ../data/train/covariates \
    --n_ext_feat     3 \
    --model_save_path ./checkpoints \
    --num_iters      10000
```

`--n_ext_feat` must equal the number of covariate sub-directories under
`--covariate_root` (here: `cov_1`, `cov_2`, `cov_3`).

### 4. Test / reconstruct the missing regions

```bash
# Baseline model
cd RFR
python run.py --test \
    --data_root ../data/test/images --mask_root ../data/test/masks \
    --model_path ./checkpoints/g_final.pth \
    --result_save_path ./results

# Covariate-guided model
cd RFR_all
python run.py --test \
    --data_root ../data/test/images --mask_root ../data/test/masks \
    --covariate_root ../data/test/covariates --n_ext_feat 3 \
    --model_path ./checkpoints/g_final.pth \
    --result_save_path ./results
```

### Command-line options (both models)

| Option | Default | Description |
|---|---|---|
| `--data_root` | `./data/train/images` | Directory of target GeoTIFF images. |
| `--mask_root` | `./data/train/masks` | Directory of missing-region masks (pixel > 0 = missing). |
| `--covariate_root` | `./data/train/covariates` | *(RFR_all only)* Directory of covariate layers, one sub-directory per covariate. |
| `--n_ext_feat` | 3 | *(RFR_all only)* Number of covariate layers. |
| `--target_size` | `128 128` | Training patch size (height width). |
| `--mask_mode` | 2 | 0 = external masks in random order, 1 = random free-form masks, 2 = external masks in fixed order. |
| `--num_iters` | 50000 / 10000 | Total training iterations. |
| `--batch_size` | 16 | Training batch size. |
| `--n_threads` | 6 | DataLoader worker threads. |
| `--finetune` | off | Fine-tune mode (BatchNorm frozen, lr = 5e-5). |
| `--test` | off | Run inference instead of training. |
| `--gpu_id` | 0 | GPU index. |
| `--model_path` | None | Checkpoint to resume (train) or load (test). |

## Outputs

* **Checkpoints** — `<model_save_path>/g_<iter>.pth` (every 1000
  iterations) and `g_final.pth`.
* **TensorBoard logs** — `<model_save_path>/tb_logs_<timestamp>/`
  (L1 loss curves; visualize with `tensorboard --logdir <model_save_path>`).
* **Reconstructed rasters** — `<result_save_path>/results/img_<n>.tif`
  (single-band GeoTIFF in the original data scale).
* *(RFR_all only)* **Metrics** — `<model_save_path>/metrics_<timestamp>.txt`
  (MAE, RMSE, R2, CCC every 5000 iterations and the average of the last
  10000 iterations) and **scatter density plots**
  `scatter_density_plot_<iter>.png` (ground truth vs. prediction).

## Methods summary

Both models are built on the Recurrent Feature Reasoning Network
(RFR-Net; Li et al., 2020, CVPR) with partial convolutions (Liu et al.,
2018, ECCV) that propagate validity masks through the network. The input
raster is refined by 20 recurrent reasoning steps with knowledge-consistent
attention, and the intermediate feature maps are aggregated with a
mask-weighted average. The generator is trained with a combined loss:

`L = 0.1 * TV + 120 * style + 0.05 * perceptual + 1 * L1(valid) + 6 * L1(hole)`

where the style and perceptual terms are computed on frozen VGG16
features. The covariate-guided variant concatenates the covariate layers
channel-wise to the input of the first partial convolution, and expands
the validity mask accordingly, so that missing regions are treated
consistently across all input layers.

## Notes for the manuscript/code review

* All comments and docstrings are in English, as required by the journal's
  code formatting guidelines.
* Hard-coded local paths from the development version were replaced by
  command-line arguments with sensible relative defaults.
* Dead/commented-out experimental code was removed; the functional
  behaviour of the training and testing pipelines is unchanged.
* The `PartialConv2d` implementation is © 2018 NVIDIA Corporation
  (BSD 3-Clause License) and retains its original license header.
* The RFR-Net architecture follows Li, J., Wang, N., Zhang, L., Du, B.,
  Tao, D. (2020). Recurrent Feature Reasoning for Image Inpainting.
  CVPR 2020.

## License

This code is released under the BSD 3-Clause License (see `LICENSE`).
Note that `modules/partialconv2d.py` is © 2018 NVIDIA Corporation
(BSD 3-Clause License) and must retain its copyright notice.

## Citation

If you use this code, please cite:

```bibtex
@article{ban2026covariate,
  title   = {Covariate-guided reconstruction of missing geochemical data:
             An application to mineral prospectivity mapping},
  author  = {Ban, Siyuan and Zhang, Zhenjie and Jian, Fuyuan and Zhou, Kexin},
  journal = {Computers \& Geosciences},
  year    = {2026},
  note    = {Manuscript}
}
```

## Contact

For questions about the code, please contact:

Zhenjie Zhang (corresponding author)
China University of Geosciences (Beijing)
E-mail: zjzhang@cugb.edu.cn
