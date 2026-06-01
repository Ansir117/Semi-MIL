import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CAMELYON16FeatureDataset(Dataset):
    """
    CAMELYON16 feature dataset.
    Expected files under feature_root:
      - training_features.npz
      - testing_features.npz
    """

    def __init__(self, feature_root, train=True, return_bag=False, max_bag_size=100):
        self.feature_root = feature_root
        self.train = train
        self.return_bag = return_bag
        self.max_bag_size = max_bag_size
        self.split = "training" if train else "testing"

        npz_path = os.path.join(feature_root, f"{self.split}_features.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Feature file not found: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        self.features = data["features"].astype(np.float32)
        self.patch_labels = data["patch_labels"].astype(np.int64)
        self.patch_has_gt = data["patch_has_gt"].astype(np.int64)
        self.slide_labels = data["slide_labels"].astype(np.int64)
        self.slide_indices = data["slide_indices"].astype(np.int64)
        self.slide_names = data["slide_names"].astype(str)

        self.num_patches = self.features.shape[0]
        self.unique_slide_ids = np.unique(self.slide_indices)
        self.num_slides = len(self.unique_slide_ids)
        self.slideid_to_patch_indices = {
            int(sid): np.where(self.slide_indices == sid)[0]
            for sid in self.unique_slide_ids
        }

        print(f"[FEAT DATA] split={self.split}, slides={self.num_slides}, patches={self.num_patches}")
        print(
            f"[FEAT DATA] labeled patches={int((self.patch_has_gt == 1).sum())}, "
            f"unlabeled patches={int((self.patch_has_gt == 0).sum())}"
        )

    def __len__(self):
        return self.num_slides if self.return_bag else self.num_patches

    def __getitem__(self, index):
        if self.return_bag:
            slide_id = int(self.unique_slide_ids[index])
            idx_patch = self.slideid_to_patch_indices[slide_id]
            if len(idx_patch) > self.max_bag_size:
                idx_patch = np.random.choice(idx_patch, self.max_bag_size, replace=False)

            bag_feat = torch.from_numpy(self.features[idx_patch])
            patch_labels = torch.from_numpy(self.patch_labels[idx_patch])
            patch_has_gt = torch.from_numpy(self.patch_has_gt[idx_patch])
            slide_label = torch.tensor(self.slide_labels[idx_patch[0]], dtype=torch.long)
            slide_name = self.slide_names[idx_patch[0]]

            return bag_feat, [patch_labels, slide_label, torch.tensor(slide_id), slide_name, patch_has_gt], index

        feat = torch.from_numpy(self.features[index])
        patch_label = torch.tensor(self.patch_labels[index], dtype=torch.long)
        slide_label = torch.tensor(self.slide_labels[index], dtype=torch.long)
        slide_idx = torch.tensor(self.slide_indices[index], dtype=torch.long)
        slide_name = self.slide_names[index]
        patch_has_gt = torch.tensor(self.patch_has_gt[index], dtype=torch.long)
        return feat, [patch_label, slide_label, slide_idx, slide_name, patch_has_gt], index


def create_ssl_split_feature(dataset, labeled_slide_ratio=0.2, labeled_patch_ratio=0.5, seed=42):
    """
    Keep the same split policy as e2e:
    - choose labeled slides by ratio (pos/neg separately),
    - sample labeled patches from labeled slides,
    - always include externally labeled patches.
    """
    if dataset.return_bag:
        raise ValueError("create_ssl_split_feature requires patch-level dataset (return_bag=False)")

    np.random.seed(seed)
    unique_slides = np.unique(dataset.slide_indices)

    pos_slides, neg_slides = [], []
    for sid in unique_slides:
        slide_mask = dataset.slide_indices == sid
        if dataset.slide_labels[slide_mask][0] == 1:
            pos_slides.append(sid)
        else:
            neg_slides.append(sid)

    num_labeled_pos = max(1, int(len(pos_slides) * labeled_slide_ratio)) if len(pos_slides) > 0 else 0
    num_labeled_neg = max(1, int(len(neg_slides) * labeled_slide_ratio)) if len(neg_slides) > 0 else 0

    labeled_pos_slides = np.random.choice(pos_slides, num_labeled_pos, replace=False) if num_labeled_pos > 0 else []
    labeled_neg_slides = np.random.choice(neg_slides, num_labeled_neg, replace=False) if num_labeled_neg > 0 else []
    labeled_slides = np.concatenate([labeled_pos_slides, labeled_neg_slides]) if (num_labeled_pos + num_labeled_neg) > 0 else np.array([])

    labeled_indices = []
    gt_indices = np.where((dataset.patch_has_gt == 1) & (dataset.patch_labels >= 0))[0]
    if len(gt_indices) > 0:
        labeled_indices.extend(gt_indices.tolist())

    for sid in labeled_slides:
        slide_mask = dataset.slide_indices == sid
        slide_patch_indices = np.where(slide_mask)[0]
        valid_indices = slide_patch_indices[dataset.patch_labels[slide_patch_indices] >= 0]
        if len(valid_indices) == 0:
            continue
        num_to_sample = max(1, int(len(valid_indices) * labeled_patch_ratio))
        sampled = np.random.choice(valid_indices, min(num_to_sample, len(valid_indices)), replace=False)
        labeled_indices.extend(sampled.tolist())

    labeled_indices = np.array(sorted(list(set(labeled_indices))), dtype=np.int64)
    all_indices = np.arange(dataset.num_patches, dtype=np.int64)
    unlabeled_indices = np.setdiff1d(all_indices, labeled_indices)

    labeled_slide_names = []
    for sid in labeled_slides:
        mask = dataset.slide_indices == sid
        labeled_slide_names.append(dataset.slide_names[mask][0])

    print(
        f"[SSL split] labeled slides={len(labeled_slides)}, "
        f"labeled patches={len(labeled_indices)}, unlabeled patches={len(unlabeled_indices)}"
    )
    return {
        "labeled_indices": labeled_indices,
        "unlabeled_indices": unlabeled_indices,
        "labeled_slides": labeled_slide_names,
        "num_labeled": len(labeled_indices),
        "num_unlabeled": len(unlabeled_indices),
    }


class SSLPatchFeatureDataset(Dataset):
    def __init__(self, base_dataset, labeled_indices, unlabeled_indices, strong_noise_std=0.05):
        if base_dataset.return_bag:
            raise ValueError("SSLPatchFeatureDataset requires patch-level dataset (return_bag=False)")
        self.base_dataset = base_dataset
        self.labeled_indices = np.array(labeled_indices, dtype=np.int64)
        self.unlabeled_indices = np.array(unlabeled_indices, dtype=np.int64)
        self.strong_noise_std = strong_noise_std

        self.all_indices = np.concatenate([self.labeled_indices, self.unlabeled_indices])
        self.is_labeled = np.concatenate([
            np.ones(len(self.labeled_indices), dtype=bool),
            np.zeros(len(self.unlabeled_indices), dtype=bool),
        ])

    def __len__(self):
        return len(self.all_indices)

    def _strong_aug(self, feat):
        if self.strong_noise_std <= 0:
            return feat
        noise = torch.randn_like(feat) * self.strong_noise_std
        return feat + noise

    def __getitem__(self, idx):
        ridx = int(self.all_indices[idx])
        is_labeled = bool(self.is_labeled[idx])
        feat = torch.from_numpy(self.base_dataset.features[ridx].astype(np.float32))
        feat_w = feat.clone()
        feat_s = feat.clone() if is_labeled else self._strong_aug(feat.clone())
        patch_label = int(self.base_dataset.patch_labels[ridx])
        slide_label = int(self.base_dataset.slide_labels[ridx])
        slide_idx = int(self.base_dataset.slide_indices[ridx])
        has_gt = int(self.base_dataset.patch_has_gt[ridx])

        return {
            "feat": feat,
            "feat_w": feat_w,
            "feat_s": feat_s,
            "patch_label": torch.tensor(patch_label, dtype=torch.long),
            "slide_label": torch.tensor(slide_label, dtype=torch.long),
            "slide_idx": torch.tensor(slide_idx, dtype=torch.long),
            "is_labeled": torch.tensor(is_labeled),
            "has_gt": torch.tensor(has_gt, dtype=torch.long),
            "index": torch.tensor(ridx, dtype=torch.long),
        }
