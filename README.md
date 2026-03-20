# PE Fusion 

This repository contains the scripts and models used in the paper *"Multimodal fusion with deep neural networks for leveraging CT imaging and electronic health record: a case-study in pulmonary embolism detection"* published on Nature Scientific Reports.[manuscript link](https://www-nature-com.stanford.idm.oclc.org/articles/s41598-020-78888-w)


## Abstract 

Recent advancements in deep learning have led to a resurgence of medical imaging and Electronic Medical Record (EMR) models for a variety of applications, including clinical decision support, automated workflow triage, clinical prediction and more. However, very few models have been developed to integrate both clinical and imaging data, despite that in routine practice clinicians rely on EMR to provide context in medical imaging interpretation. In this study, we developed and compared different multimodal fusion model architectures that are capable of utilizing both pixel data from volumetric Computed Tomography Pulmonary Angiography scans and clinical patient data from the EMR to automatically classify Pulmonary Embolism (PE) cases. The best performing multimodality model is a late fusion model that achieves an AUROC of 0.947 [95% CI: 0.946–0.948] on the entire held-out test set, outperforming imaging-only and EMR-only single modality models.

## Methods

In this study, we separately trained an imaging-only model [PENet](https://rdcu.be/b3Lll) and 7 EMR-only neural network models for each modality. These single modality models not only serve as baselines for performance comparison, they also provide different inputs and components for different fusion models. A total of 7 fusion architectures were implemented and compared:

![](./img/fusion_architectures.png)

## Results

Model performance on the held-out testset with 95% confidence interval using probability threshold that maximizes both sensitivity and specificity on the validation dataset. 

|                   |Imaging model      |EMR model          |Late elastic average Fusion|
|-------------------|-------------------|-------------------|---------------------------|
|Operating threshold|0.625              |0.63               |0.448                      |
|Accuracy           |0.687 [0.685–0.689]|0.834 [0.832–0.835]|0.885 [0.884–0.886]        |
|AUROC              |0.791 [0.788–0.793]|0.911 [0.910–0.913]|0.947 [0.946–0.948]        |
|Specificity        |0.862 [0.860–0.865]|0.875 [0.872–0.877]|0.902 [0.9–0.904]          |
|Sensitivity        |0.559 [0.557–0.562]|0.804 [0.801–0.806]|0.873 [0.871–0.875]        |
|PPV                |0.848 [0.846–0.851]|0.898 [0.896–0.899]|0.924 [0.923–0.926         |
|NPV                |0.588 [0.585–0.590]|0.765 [0.761–0.767]|0.838 [0.835–0.84]         |


## Setup

### Prerequisites

* Python ≥ 3.7
* [conda](https://docs.conda.io/en/latest/) (recommended)

### Installation

```bash
conda env create -f environment.yml
conda activate pe_fusion
```

### Data directory

The code expects a data root at `/data/fusion` (configurable in
`constants/constants.py`).  The directory must contain:

```
/data/fusion/
├── emr_data/          # Raw EMR CSV files (All.csv, Demographics.csv, …)
├── vision_feature/    # Pre-extracted imaging features (vision.pickle)
├── mappings/          # Train/val/test split and label pickle files
├── parsed_data/       # Created by the pre-processing step (see below)
├── logs/              # Training logs (created automatically)
├── ckpt/              # Model checkpoints (created automatically)
└── results/           # Prediction results (created automatically)
```


## Repository Structure

```
pe_fusion/
├── constants/              # Global paths and hyper-parameter grids
│   ├── constants.py        # Directory paths and column-name constants
│   └── gridsearch.py       # Elastic-net grid-search parameter space
├── dataset/                # PyTorch Dataset and DataLoader wrappers
│   ├── emr_dataset.py      # Single-modality EMR dataset
│   └── fusion_dataset.py   # Multi-modality fusion dataset
├── lightning/              # PyTorch Lightning training/evaluation wrapper
│   └── lightning_model.py  # LightningModule (train, val, test loops)
├── models/                 # Model definitions
│   ├── fcnn.py             # Fully connected neural network (FCNN)
│   ├── joint_fusion.py     # Joint (early) fusion model
│   ├── late_fusion.py      # Late average fusion script
│   └── elastic.py          # Elastic-net logistic regression baseline
├── preprocess/             # Data pre-processing scripts
│   ├── create_dataset.py   # Parse raw CSVs → train/val/test pickle files
│   └── remove_subseg.py    # Filter out subsegmental PE cases
├── train.py                # Main training entry-point
├── test.py                 # Evaluation on a saved checkpoint
├── train.sh                # Example training shell script
├── test.sh                 # Example testing shell script
└── sweep.yaml              # Weights & Biases hyper-parameter sweep config
```


## Usage

### 1 — Pre-process EMR data

```bash
python preprocess/create_dataset.py
```

This reads the raw CSV files from `emr_data/`, applies z-score normalisation
and splits the data into `train`, `val`, and `test` pickle files under
`parsed_data/`.

Optionally, remove subsegmental PE cases:

```bash
python preprocess/remove_subseg.py
```

### 2 — Train a model

Single EMR modality (e.g. Demographics):

```bash
python train.py \
    --data_type Demographics \
    --num_neurons 512 --num_hidden 2 \
    --activation ReLU --init_method kaiming \
    --dropout_prob 0.2 --lr 1e-4 \
    --optimizer adam --batch_size 64 \
    --max_epochs 50 --gpus 1
```

Joint fusion (all EMR modalities + imaging features):

```bash
python train.py \
    --data_type JointAll \
    --num_neurons 512 --num_hidden 2 \
    --activation ReLU --init_method kaiming \
    --dropout_prob 0.2 --lr 1e-4 \
    --optimizer adam --batch_size 64 \
    --max_epochs 50 --gpus 1
```

The `data_type` argument accepts:

| Value | Description |
|-------|-------------|
| `Demographics`, `ICD`, `LABS`, `Vitals`, `INP_MED`, `OUT_MED`, `All` | Single EMR modality |
| `Vision` | Imaging features only |
| `JointSeparate` | Aggregated EMR (`All`) + imaging |
| `JointAll` | All individual EMR modalities + imaging |

### 3 — Evaluate a checkpoint

```bash
python test.py --checkpoint_path /data/fusion/ckpt/<experiment>/<ckpt>.ckpt
```

Results are saved to `RESULTS_DIR/<experiment_name>/results.csv`.

### 4 — Late fusion (probability averaging)

After generating individual model result files, combine them:

```bash
python models/late_fusion.py \
    --result_paths results/emr_model/results.csv,results/imaging_model/results.csv \
    --late_fusion_name late_all
```

### 5 — Elastic-net baseline

```bash
python models/elastic.py
```


## Citation
Huang, SC., Pareek, A., Zamanian, R. et al. Multimodal fusion with deep neural networks for leveraging CT imaging and electronic health record: a case-study in pulmonary embolism detection. Sci Rep 10, 22147 (2020). https://doi-org.stanford.idm.oclc.org/10.1038/s41598-020-78888-w

## Data Availability 
The datasets generated and analyzed during the study are not currently publicly available due to HIPAA compliance agreement but are available from the corresponding author on reasonable request.