# Igformer - Antibody Design Framework

Igformer emphasizes interface calculation with higher dockq accuracy for antibody design.

## Project Structure

```
igformer/
├── Igformer/                   # Igformer model implementation
│   └── models1/
│       ├── models/igformer/    # Igformer model implementation
│       └── trainer/            # Trainer implementation
├── scripts/train/               # Training scripts
│   ├── train_igformer.sh      # Igformer training launcher script
│   └── configs/                # Configuration files
│       └── single_cdr_design_igformer.json
├── train2.py                   # Main training entry point
├── data/                       # Data processing
├── all_data/                   # Dataset (RAbD dataset)
├── test_all_checkpoints.py     # Igformer testing script
├── utils/                     # Utility functions
└── README.md                  # This file
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+
- 8x A100 40GB GPUs (recommended for optimal performance)
- Other dependencies see `env.yml`

## Hardware Performance

- **GPU Configuration**: 8x NVIDIA A100 40GB
- **Training Speed**: Approximately 1 minute per epoch
- **Memory Usage**: Optimized for multi-GPU distributed training

## Quick Start

### 1. Training Igformer Model

**Igformer Model Training:**
```bash
GPU="0,1,2,3,4,5,6,7" nohup bash scripts/train/train_igformer.sh scripts/train/configs/single_cdr_design_igformer.json > training_igformer.log 2>&1 &
```

**Parameter Description:**
- `GPU`: Specify GPU devices to use
- `train_igformer.sh`: Igformer training launcher script
- `train2.py`: Main training entry point (called by train_igformer.sh)
- `single_cdr_design_igformer.json`: Igformer configuration file
- `training_igformer.log`: Training log output file

### 2. Testing Igformer Model

**Test Igformer Checkpoints:**
```bash
python test_all_checkpoints.py --checkpoint_dir my_checkpoints/models_igformer/version_7/checkpoint/ --test_set all_data/RAbD/test_fixed.json --save_dir results --batch_size 1 --gpu 6
```

**Download Pre-trained Checkpoints:**
```bash
# Download the compressed checkpoint file
wget [checkpoint_url]/igformer_checkpoints_version_7.tar.gz

# Extract the checkpoints
tar -xzf igformer_checkpoints_version_7.tar.gz
```

**Parameter Description:**
- `--checkpoint_dir`: Checkpoint file directory
- `--test_set`: Test dataset path
- `--save_dir`: Results save directory
- `--batch_size`: Batch size
- `--gpu`: GPU device to use

## Configuration Files

### Igformer Training Configuration (single_cdr_design_igformer.json)

```json
{
  "train_set": "all_data/RAbD/train_fixed.json",
  "valid_set": "all_data/RAbD/valid_fixed.json",
  "save_dir": "my_checkpoints/models_igformer",
  "cdr": "H3",
  "max_epoch": 150,
  "model_type": "igformer",
  "batch_size": 32,
  "lr": 0.0005,
  "embed_dim": 64,
  "hidden_size": 128,
  "k_neighbors": 9,
  "n_layers": 3,
  "iter_round": 3,
  "bind_dist_cutoff": 6.6,
  "paratope": "H3",
  "backbone_only": false,
  "fix_channel_weights": false,
  "no_pred_edge_dist": false,
  "no_memory": false
}
```

## Evaluation Metrics

- **AAR**: Amino Acid Recovery
- **CAAR**: Contact Amino Acid Recovery  
- **RMSD**: Root Mean Square Deviation
- **TMscore**: Template Modeling score
- **DockQ**: Docking Quality score

## Training Script Details

### train2.py - Main Training Entry Point
- **Purpose**: Main Python training script for Igformer model
- **Features**:
  - Multi-GPU distributed training support
  - Automatic model and trainer selection based on `--model_type`
  - Configurable hyperparameters via command line arguments
  - Automatic checkpoint saving and validation
- **Key Parameters**:
  - `--model_type`: Model type (currently supports 'igformer')
  - `--train_set`/`--valid_set`: Dataset paths
  - `--save_dir`: Checkpoint save directory
  - `--batch_size`: Training batch size
  - `--max_epoch`: Maximum training epochs
  - `--gpus`: GPU devices to use
  - `--lr`/`--final_lr`: Learning rate and final learning rate for exponential decay
- **Usage**: Called by `train_igformer.sh` or can be run directly

### train_igformer.sh - Training Launcher Script
- **Purpose**: Shell script wrapper for distributed training
- **Features**:
  - Multi-GPU setup and configuration
  - JSON config file parsing
  - Distributed training with torchrun
  - Environment variable handling

## Dataset and Checkpoints

### Dataset
- **Location**: `all_data/` directory contains the RAbD (Rosetta Antibody Database) dataset
- **Training Data**: `all_data/RAbD/train_fixed.json`
- **Validation Data**: `all_data/RAbD/valid_fixed.json` 
- **Test Data**: `all_data/RAbD/test_fixed.json`

### Pre-trained Checkpoints
- **Location**: `my_checkpoints/models_igformer/version_7/checkpoint/`
- **Compressed File**: `igformer_checkpoints_version_7.tar.gz` (201MB)
- **Contents**: 10 best model checkpoints from training epochs 96-120
- **Best Checkpoint**: `epoch108_step20383.ckpt` (DockQ score: 0.4852, TMscore: 0.9738)
- **Test Results**: Available in `epoch108_step20383_results.txt`

## Notes

1. Ensure dataset paths are correctly configured
2. Large GPU memory requirements, recommend using multi-GPU training
3. Best checkpoints are automatically saved during training
4. Ensure checkpoint paths exist when testing
5. All training scripts support multi-GPU distributed training using the specified GPU list
6. Download the compressed checkpoint file for easy deployment and testing

## Model Performance Results

### Best Checkpoint Performance (epoch108_step20383.ckpt)

The best performing checkpoint has been tested and shows excellent results:

**Performance Metrics:**
- **AAR (Amino Acid Recovery)**: 0.4303
- **CAAR (Contact Amino Acid Recovery)**: 0.2965
- **RMSD(CA)**: 20.3686
- **RMSD(CA) CDRH3**: 6.6141
- **TMscore**: 0.9738
- **LDDT**: 0.0000
- **DockQ**: 0.4852

**Model Configuration:**
- Model Type: Igformer
- CDR: H3
- Hidden Size: 128
- Embed Dim: 64
- Layers: 3
- Iter Rounds: 3
- Batch Size: 32
- Learning Rate: 0.0005

**Interpretation:**
- **DockQ score of 0.4852** indicates good docking quality
- **TMscore of 0.9738** shows excellent structural similarity
- **AAR of 0.4303** demonstrates reasonable amino acid recovery
- **CAAR of 0.2965** indicates contact amino acid recovery performance

**Direct Inference Usage:**
```bash
# Use the best checkpoint for inference
python test_all_checkpoints.py --checkpoint_dir my_checkpoints/models_igformer/version_7/checkpoint/ --test_set your_test_data.json --save_dir results --batch_size 1 --gpu 0
```

## Troubleshooting

1. **Permission Issues**: Ensure executable files in evaluation directory have execution permissions
2. **Path Issues**: Check that all relative paths are correct
3. **GPU Memory**: If encountering OOM, reduce batch_size or use fewer GPUs
4. **Checkpoint Loading**: Make sure the checkpoint directory path is correct and contains `.ckpt` files for the Igformer model.


## Dataset

You can download the RAbD dataset and place it under `/all_data` directory.

**RAbD Dataset Download:**
- **Source**: Baidu Cloud Drive
- **Link**: https://pan.baidu.com/s/1PVSOj61BpST-GCdRQQKJnQ?pwd=jihw
- **Extraction Code**: jihw
- **Note**: Shared by Baidu Cloud Drive Super VIP v6

**Dataset Structure:**
```
all_data/
├── RAbD/
│   ├── train_fixed.json          # Training dataset (JSON format)
│   ├── valid_fixed.json          # Validation dataset (JSON format)
│   ├── test_fixed.json           # Test dataset (JSON format)
│   ├── train_fixed_processed/    # Preprocessed training data (PKL format)
│   ├── valid_fixed_processed/    # Preprocessed validation data (PKL format)
│   └── test_fixed_processed/     # Preprocessed test data (PKL format)
└── pdb/                          # PDB structure files
```

**File Format Description:**
- **JSON files**: Raw dataset files containing protein complex information, sequences, and metadata
- **Processed directories**: Contain preprocessed PKL files for faster data loading during training
- **PDB files**: Protein structure files used for structural analysis and evaluation
