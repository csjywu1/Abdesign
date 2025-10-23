# AbDesign - Antibody Design Framework

A comprehensive antibody design framework featuring multiple state-of-the-art models for antibody-antigen complex prediction and optimization.

**Key Features:**
- **Abd Model**: Enhanced interface calculation with higher DockQ accuracy (renamed from abdesign)
- **Igformer Model**: Graph transformer-based approach with SGFormer implementation
- **IGD Model**: Iterative graph-based design methodology
- **Task Models**: Specialized variants (Task1v0, Task1v1, Task2v0) for different design scenarios
- **dyMEAN Model**: Dynamic mean field approach for antibody design

## Project Structure

```
abdesign/
├── abd/                        # Core model code (renamed from abdesign)
│   └── models1/
│       ├── models/abd/         # abd model implementation
│       └── trainer/            # Trainer implementation
├── IGD/                        # IGD model implementation
├── task1v0/                    # Task1v0 model implementation
├── task1v1/                    # Task1v1 model implementation
├── task2v0/                    # Task2v0 model implementation
├── dyMEAN-main/                # dyMEAN model implementation
├── Igformer/                   # Igformer model implementation
├── scripts/train/               # Training scripts
│   ├── train_abd.sh           # Training launcher script (renamed from train_abdesign.sh)
│   ├── train_igformer.sh      # Igformer training launcher script
│   ├── train_igd.sh           # IGD training launcher script
│   ├── train_task1v0.sh       # Task1v0 training launcher script
│   ├── train_task1v1.sh       # Task1v1 training launcher script
│   ├── train_task2v0.sh       # Task2v0 training launcher script
│   └── configs/                # Configuration files
├── train_dymean.py            # dyMEAN training entry point
├── test_dymean.py             # dyMEAN testing entry point
├── data/                       # Data processing
├── evaluation/                 # Evaluation tools
├── all_data/                   # Dataset
├── test_*.py                   # Model-specific testing scripts
├── utils/                     # Utility functions
└── README.md                  # This file
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+
- Other dependencies see `env.yml`

## Quick Start

### 1. Training Model

**Abd Model Training:**
```bash
GPU="0,1,2,3,4,5,6,7" nohup bash scripts/train/train_abd.sh scripts/train/configs/single_cdr_design_abd.json > training.log 2>&1 &
```

**Igformer Model Training:**
```bash
GPU="0,1,2,3,4,5,6,7" nohup bash scripts/train/train_igformer.sh scripts/train/configs/single_cdr_design_igformer.json > training_igformer.log 2>&1 &
```

**IGD Model Training:**
```bash
GPU="0,1,2,3,4,5,6,7" nohup bash scripts/train/train_igd.sh scripts/train/configs/single_cdr_design_igd.json > training_igd.log 2>&1 &
```

**Task1v0 Model Training:**
```bash
GPU="0,1,2,3,4,5,6,7" nohup bash scripts/train/train_task1v0.sh scripts/train/configs/single_cdr_design_task1v0.json > training_task1v0.log 2>&1 &
```

**Task1v1 Model Training:**
```bash
GPU="0,1,2,3,4,5,6,7" nohup bash scripts/train/train_task1v1.sh scripts/train/configs/single_cdr_design_task1v1.json > training_task1v1.log 2>&1 &
```

**Task2v0 Model Training:**
```bash
GPU="0,1,2,3,4,5,6,7" nohup bash scripts/train/train_task2v0.sh scripts/train/configs/single_cdr_design_task2v0.json > training_task2v0.log 2>&1 &
```

**dyMEAN Model Training:**
```bash
GPU="0,1,2,3,4,5,6,7" nohup bash scripts/train/train_dymean.sh scripts/train/configs/single_cdr_design_dymean.json > training_dymean.log 2>&1 &
```

**Parameter Description:**
- `GPU`: Specify GPU devices to use
- `train_abd.sh`: Abd training launcher script (renamed from train_abdesign.sh)
- `train_igformer.sh`: Igformer training launcher script
- `train_igd.sh`: IGD training launcher script
- `train_task1v0.sh`: Task1v0 training launcher script
- `train_task1v1.sh`: Task1v1 training launcher script
- `train_task2v0.sh`: Task2v0 training launcher script
- `train_dymean.sh`: dyMEAN training launcher script
- `single_cdr_design_abd.json`: Abd configuration file (renamed from single_cdr_design_abdesign.json)
- `single_cdr_design_igformer.json`: Igformer configuration file
- `single_cdr_design_igd.json`: IGD configuration file
- `single_cdr_design_task1v0.json`: Task1v0 configuration file
- `single_cdr_design_task1v1.json`: Task1v1 configuration file
- `single_cdr_design_task2v0.json`: Task2v0 configuration file
- `single_cdr_design_dymean.json`: dyMEAN configuration file
- `training.log`: Training log output file

### 2. Testing Model

**Test Abd Checkpoints:**
```bash
python test_abd_checkpoints.py --checkpoint_dir my_checkpoints/models_abd/version_7/checkpoint/ --test_set all_data/RAbD/test_fixed.json --save_dir results --batch_size 2 --gpu 6
```

**Test Igformer Checkpoints:**
```bash
python test_all_checkpoints.py --checkpoint_dir my_checkpoints/models_igformer/version_0/checkpoint/ --test_set all_data/RAbD/test_fixed.json --save_dir results --batch_size 1 --gpu 6
```

**Test IGD Checkpoints:**
```bash
python test_igd_checkpoints.py --checkpoint_dir my_checkpoints/models_igd/version_1/checkpoint/ --test_set all_data/RAbD/test_fixed.json --save_dir results --batch_size 1 --gpu 6
```

**Test Task1v0 Checkpoints:**
```bash
python test_task1v0_checkpoints.py --checkpoint_dir my_checkpoints/models_task1v0/version_0/checkpoint/ --test_set all_data/RAbD/test_fixed.json --save_dir results --batch_size 1 --gpu 6
```

**Test Task1v1 Checkpoints:**
```bash
python test_task1v1_checkpoints.py --checkpoint_dir my_checkpoints/models_task1v1/version_2/checkpoint/ --test_set all_data/RAbD/test_fixed.json --save_dir results --batch_size 1 --gpu 6
```

**Test Task2v0 Checkpoints:**
```bash
python test_task2v0_checkpoints.py --checkpoint_dir my_checkpoints/models_task2v0/version_0/checkpoint/ --test_set all_data/RAbD/test_fixed.json --save_dir results --batch_size 1 --gpu 6
```

**Test dyMEAN Checkpoints:**
```bash
python test_dymean.py --checkpoint_dir my_checkpoints/models_dymean/version_0/checkpoint/ --test_set all_data/RAbD/test_fixed.json --save_dir results --batch_size 1 --gpu 6
```

**Testing Scripts:**

Each model has its own dedicated testing script to avoid module import conflicts and ensure proper model loading:

- `test_all_checkpoints.py`: For testing **Igformer** model checkpoints (legacy script, maintained for compatibility)
- `test_abd_checkpoints.py`: For testing **Abd** model checkpoints  
- `test_igd_checkpoints.py`: For testing **IGD** model checkpoints
- `test_task1v0_checkpoints.py`: For testing **Task1v0** model checkpoints
- `test_task1v1_checkpoints.py`: For testing **Task1v1** model checkpoints
- `test_task2v0_checkpoints.py`: For testing **Task2v0** model checkpoints
- `test_dymean.py`: For testing **dyMEAN** model checkpoints

**Note**: Each testing script has isolated import paths to prevent conflicts when loading different model architectures. Use the appropriate script for your specific model type.

**Parameter Description:**
- `--checkpoint_dir`: Checkpoint file directory
- `--test_set`: Test dataset path
- `--save_dir`: Results save directory
- `--batch_size`: Batch size
- `--gpu`: GPU device to use

## Configuration Files

### Abd Training Configuration (single_cdr_design_abd.json)

```json
{
  "train_set": "all_data/RAbD/train_fixed.json",
  "valid_set": "all_data/RAbD/valid_fixed.json",
  "save_dir": "my_checkpoints/models_abd",
  "cdr": "H3",
  "max_epoch": 150,
  "model_type": "abd",
  "batch_size": 32,
  "lr": 0.0005,
  "embed_dim": 64,
  "hidden_size": 128,
  "k_neighbors": 9,
  "n_layers": 3,
  "iter_round": 3,
  "bind_dist_cutoff": 6.6,
  "paratope": "H3"
}
```

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

### IGD Training Configuration (single_cdr_design_igd.json)

```json
{
  "train_set": "all_data/RAbD/train_fixed.json",
  "valid_set": "all_data/RAbD/valid_fixed.json",
  "save_dir": "my_checkpoints/models_igd",
  "cdr": "H3",
  "max_epoch": 150,
  "model_type": "igd",
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
  "no_memory": false,
  "struct_only": false,
  "sequence_loss_weight": 1.0,
  "structure_loss_weight": 1.0,
  "docking_loss_weight": 1.0,
  "pdev_loss_weight": 1.0
}
```

### Task1v0 Training Configuration (single_cdr_design_task1v0.json)

```json
{
  "train_set": "all_data/RAbD/train_fixed.json",
  "valid_set": "all_data/RAbD/valid_fixed.json",
  "save_dir": "my_checkpoints/models_task1v0",
  "cdr": "H3",
  "max_epoch": 150,
  "model_type": "task1v0",
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
  "no_memory": false,
  "struct_only": false,
  "sequence_loss_weight": 1.0,
  "structure_loss_weight": 1.0,
  "docking_loss_weight": 1.0,
  "pdev_loss_weight": 1.0
}
```

### Task1v1 Training Configuration (single_cdr_design_task1v1.json)

```json
{
  "train_set": "all_data/RAbD/train_fixed.json",
  "valid_set": "all_data/RAbD/valid_fixed.json",
  "save_dir": "my_checkpoints/models_task1v1",
  "cdr": "H3",
  "max_epoch": 150,
  "model_type": "task1v1",
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
  "no_memory": false,
  "struct_only": false,
  "sequence_loss_weight": 1.0,
  "structure_loss_weight": 1.0,
  "docking_loss_weight": 1.0,
  "pdev_loss_weight": 1.0
}
```

### Task2v0 Training Configuration (single_cdr_design_task2v0.json)

```json
{
  "train_set": "all_data/RAbD/train_fixed.json",
  "valid_set": "all_data/RAbD/valid_fixed.json",
  "save_dir": "my_checkpoints/models_task2v0",
  "cdr": "H3",
  "max_epoch": 150,
  "model_type": "task2v0",
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
  "no_memory": false,
  "struct_only": false,
  "sequence_loss_weight": 1.0,
  "structure_loss_weight": 1.0,
  "docking_loss_weight": 1.0,
  "pdev_loss_weight": 1.0
}
```

### dyMEAN Training Configuration (single_cdr_design_dymean.json)

```json
{
  "train_set": "all_data/RAbD/train_fixed.json",
  "valid_set": "all_data/RAbD/valid_fixed.json",
  "save_dir": "my_checkpoints/models_dymean",
  "cdr": "H3",
  "max_epoch": 150,
  "save_topk": 1,
  "model_type": "dymean",
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
  "no_memory": false,
  "struct_only": false,
  "sequence_loss_weight": 1.0,
  "structure_loss_weight": 1.0,
  "docking_loss_weight": 1.0,
  "pdev_loss_weight": 1.0
}
```

## Evaluation Metrics

- **AAR**: Amino Acid Recovery
- **CAAR**: Contact Amino Acid Recovery  
- **RMSD**: Root Mean Square Deviation
- **TMscore**: Template Modeling score
- **DockQ**: Docking Quality score

## Notes

1. Ensure dataset paths are correctly configured
2. Large GPU memory requirements, recommend using multi-GPU training
3. Best checkpoints are automatically saved during training
4. Ensure checkpoint paths exist when testing
5. Each model type requires its specific testing script to avoid import path conflicts and ensure proper model loading
6. All training scripts support multi-GPU distributed training using the specified GPU list
7. dyMEAN model uses a different directory structure (dyMEAN-main/) and has its own training/testing scripts

## Troubleshooting

1. **Permission Issues**: Ensure executable files in evaluation directory have execution permissions
2. **Path Issues**: Check that all relative paths are correct
3. **GPU Memory**: If encountering OOM, reduce batch_size or use fewer GPUs
4. **Model Import Issues**: If you encounter import errors during testing, ensure you're using the correct testing script for your model type. Each model has its dedicated testing script to avoid module conflicts.
5. **Checkpoint Loading**: Make sure the checkpoint directory path is correct and contains `.ckpt` files for the appropriate model.
6. **dyMEAN Specific Issues**: 
   - Ensure dyMEAN-main/models1/ directory structure is intact
   - Check that train_dymean.py and test_dymean.py are in the project root directory
   - Verify dyMEAN-main/scripts/train/train_dymean.sh has correct paths


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
