# Semi-WENO (WENO + InstanT-style patch SSL)

This folder is a ready-to-upload standalone project based on `WENO-main`, with InstanT-style semi-supervised patch classifier training integrated into the cervical training pipeline.

## Key modified files

- `Datasets_loader/dataset_CervicalCancer.py`
  - Add optional `patch_label_file` support.
  - Return `patch_label_has_gt` flag for each patch.
- `train_Cervical_BagDistillationDSMIL_SharedEnc_Similarity_StuFilterSmoothed_DropPos.py`
  - Add `--student_ssl_mode {weno,instant}`.
  - In `instant` mode, mix real patch labels with confident pseudo labels.
  - Add args:
    - `--patch_label_file`
    - `--inst_weight_supervised`
    - `--inst_conf_threshold`
- `experiments/instant_cervical/README.md`
- `experiments/instant_cervical/patch_labels_template.csv`

## Run example

```bash
python3 train_Cervical_BagDistillationDSMIL_SharedEnc_Similarity_StuFilterSmoothed_DropPos.py \
  --student_ssl_mode instant \
  --patch_label_file experiments/instant_cervical/patch_labels_template.csv \
  --inst_weight_supervised 0.8 \
  --inst_conf_threshold 0.6 \
  --epochs 300
```

## CAMELYON16 feature workflow (ResNet50)

### 1) Extract features first

```bash
cd Semi-Weno
python3 extract_features_resnet50.py \
  --data_dir /home/xiaoyuan/Data3/CAMELYON16 \
  --output_dir /home/xiaoyuan/Data3/CAMELYON16/features_resnet50 \
  --split both
```

This generates:
- `/home/xiaoyuan/Data3/CAMELYON16/features_resnet50/training_features.npz`
- `/home/xiaoyuan/Data3/CAMELYON16/features_resnet50/testing_features.npz`

### 2) Train Semi-WENO on pre-extracted features

```bash
cd Semi-Weno
python3 train_feat.py \
  --feature_root /home/xiaoyuan/Data3/CAMELYON16/features_resnet50 \
  --feature_input_dim 2048 \
  --feature_dim 512
```

## Notes

- Keep your dataset layout unchanged as expected by the original WENO code.
- Fill `patch_labels_template.csv` with your real patch labels before formal experiments.
