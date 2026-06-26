# WhiteBox-Cervical

Code release for **Compact and Structurally Transparent Cervical Cytology with Geometry-Driven Features and Closed-Form Attention**.  
The model derives attention from explicit pathological concepts and uses it solely for interpretable feature sampling.


---

### Environment

Create and activate the environment from the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate pytorch_1.11.0

```

----------

### Dataset

We evaluate on **DSCC**, **Herlev**, and **SIPaKMeD** with 5-fold cross-validation.

**DSCC (3 classes)**

```
datasets/
└── DSCC/
    └── splited/
        └── seed_0_5fold/
            ├── 0/
            │   ├── train/
            │   │   ├── ASC/
            │   │   ├── NORMAL/
            │   │   └── SIL/
            │   └── validation/
            │       ├── ASC/
            │       ├── NORMAL/
            │       └── SIL/
            ├── 1/
            ├── 2/
            ├── 3/
            └── 4/

```

**Herlev (7 classes)**

```
datasets/
└── Herlev/
    └── splited/
        └── seed_0_5fold/
            ├── 0/
            │   ├── train/
            │   │   ├── carcinoma_in_situ/
            │   │   ├── light_dysplastic/
            │   │   ├── moderate_dysplastic/
            │   │   ├── normal_columnar/
            │   │   ├── normal_intermediate/
            │   │   ├── normal_superficiel/
            │   │   └── severe_dysplastic/
            │   └── validation/
            │       ├── carcinoma_in_situ/
            │       ├── light_dysplastic/
            │       ├── moderate_dysplastic/
            │       ├── normal_columnar/
            │       ├── normal_intermediate/
            │       ├── normal_superficiel/
            │       └── severe_dysplastic/
            ├── 1/
            ├── 2/
            ├── 3/
            └── 4/

```

**SIPaKMeD (5 classes)**

```
datasets/
└── SIP/
    └── splited/
        └── seed_0_5fold/
            ├── 0/
            │   ├── train/
            │   │   ├── im_Dyskeratotic/
            │   │   ├── im_Koilocytotic/
            │   │   ├── im_Metaplastic/
            │   │   ├── im_Parabasal/
            │   │   └── im_Superficial-Intermediate/
            │   └── validation/
            │       ├── im_Dyskeratotic/
            │       ├── im_Koilocytotic/
            │       ├── im_Metaplastic/
            │       ├── im_Parabasal/
            │       └── im_Superficial-Intermediate/
            ├── 1/
            ├── 2/
            ├── 3/
            └── 4/

```

**Split selection**

-   `--d {d|h|s}` maps to **DSCC | Herlev | SIPaKMeD**
    
-   `--fold {0..4}` selects the fold index
    


### Training

Run the unified training script (`train.py`).  
Defaults: epochs 300, batch size 16, cosine LR schedule starting at 0.004, SGD with momentum 0.9 and weight decay 5e-4.  
Images are resized to 256, randomly cropped to 224 with horizontal flip for training; center crop for evaluation.

**Examples**

```bash
# DSCC, fold 0
python train.py --d d --fold 0 --epochs 300 --seed 0

# Herlev, fold 3
python train.py --d h --fold 3 --epochs 300 --seed 0

# SIPaKMeD, fold 4
python train.py --d s --fold 4 --epochs 300 --seed 0

```

Outputs (logs and best checkpoint) are saved under:

```
results/classification/{script_name}_{DATASET}_seed_{seed}_fold_{fold}_input_size_224_lr_0.004_pretrained_{0|1}_epochs_{E}

```
