# InstanT-style Patch Branch for WENO (Cervical)

This experiment keeps your original WENO path and only adds configurable supervised patch signals into the student branch.

## 1) What changed

- `Datasets_loader/dataset_CervicalCancer.py`
  - Added optional `patch_label_file` loading (`csv` / `tsv`).
  - Supports keys as:
    - absolute patch path
    - `slide_name/patch_name`
    - patch file name only
  - Returns an extra flag `patch_label_has_gt` in `label` (`0/1`).
- `train_Cervical_BagDistillationDSMIL_SharedEnc_Similarity_StuFilterSmoothed_DropPos.py`
  - Added `--student_ssl_mode {weno,instant}`.
  - `instant` mode mixes:
    - real patch labels (strong supervision)
    - teacher pseudo labels (only confident ones)
  - Added args:
    - `--patch_label_file`
    - `--inst_weight_supervised`
    - `--inst_conf_threshold`

## 2) Label file format

Use `patch_labels_template.csv` in this folder as the template:

```text
patch_path,label
slideA/patch_0001.png,1
slideA/patch_0002.png,0
```

`label` must be `0` or `1`.

## 3) Run example

From `WENO-main`:

```bash
python3 train_Cervical_BagDistillationDSMIL_SharedEnc_Similarity_StuFilterSmoothed_DropPos.py \
  --student_ssl_mode instant \
  --patch_label_file experiments/instant_cervical/patch_labels_template.csv \
  --inst_weight_supervised 0.8 \
  --inst_conf_threshold 0.6 \
  --epochs 300
```

If you want the original WENO student branch, set:

```bash
--student_ssl_mode weno
```
