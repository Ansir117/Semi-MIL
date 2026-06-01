import argparse
import csv
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


def infer_slide_label(slide_name):
    name = slide_name.lower()
    if "pos" in name or "tumor" in name or slide_name.endswith("_1"):
        return 1
    return 0


def infer_patch_label(patch_name, slide_label):
    base = os.path.splitext(patch_name)[0].lower()
    if "_pos" in base or base.endswith("_1"):
        return 1
    if "_neg" in base or base.endswith("_0"):
        return 0
    return 0 if slide_label == 0 else -1


def load_patch_label_map(patch_label_file):
    if patch_label_file is None or patch_label_file == "":
        return {}
    if not os.path.exists(patch_label_file):
        raise FileNotFoundError(f"patch_label_file not found: {patch_label_file}")
    label_map = {}
    with open(patch_label_file, "r", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        delimiter = "," if sample.count(",") >= sample.count("\t") else "\t"
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) < 2:
                continue
            key, val = row[0].strip(), row[1].strip()
            if key == "" or val == "" or val.lower() in ["label", "patch_label"]:
                continue
            label_map[os.path.normpath(key)] = int(float(val))
    return label_map


class CamelyonPatchImageDataset(Dataset):
    def __init__(self, data_dir, split="training", patch_label_file=""):
        self.data_root = os.path.join(data_dir, split)
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"Split directory not found: {self.data_root}")

        self.label_map = load_patch_label_map(patch_label_file)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.samples = []
        slide_dirs = sorted([d for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, d))])
        for slide_idx, slide_name in enumerate(slide_dirs):
            slide_dir = os.path.join(self.data_root, slide_name)
            slide_label = infer_slide_label(slide_name)
            img_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.JPG", "*.JPEG", "*.PNG", "*.TIF"]:
                img_files.extend(glob.glob(os.path.join(slide_dir, ext)))
            img_files = sorted(img_files)
            for path in img_files:
                patch_name = os.path.basename(path)
                patch_label = infer_patch_label(patch_name, slide_label)
                patch_has_gt = 1 if patch_label >= 0 else 0

                rel_key = os.path.join(slide_name, patch_name)
                keys = [os.path.normpath(path), os.path.normpath(rel_key), os.path.normpath(patch_name)]
                for k in keys:
                    if k in self.label_map:
                        patch_label = int(self.label_map[k])
                        patch_has_gt = 1
                        break

                self.samples.append({
                    "path": path,
                    "patch_label": patch_label,
                    "patch_has_gt": patch_has_gt,
                    "slide_label": slide_label,
                    "slide_idx": slide_idx,
                    "slide_name": slide_name,
                })

        print(f"[{split}] slides={len(slide_dirs)}, patches={len(self.samples)}")
        gt_count = sum(s["patch_has_gt"] for s in self.samples)
        print(f"[{split}] labeled patches={gt_count}, unlabeled patches={len(self.samples) - gt_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        img = self.transform(img)
        return img, s["patch_label"], s["patch_has_gt"], s["slide_label"], s["slide_idx"], s["slide_name"]


def build_resnet50_feature_extractor(pretrained=True):
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        net = models.resnet50(weights=weights)
    else:
        net = models.resnet50(weights=None)
    feat_extractor = nn.Sequential(*list(net.children())[:-1])  # Nx2048x1x1
    return feat_extractor


def try_load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    clean_dict = {}
    model_dict = model.state_dict()
    for k, v in state_dict.items():
        nk = k
        for prefix in ["module.", "encoder.", "backbone."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        if nk in model_dict and model_dict[nk].shape == v.shape:
            clean_dict[nk] = v
        elif f"0.{nk}" in model_dict and model_dict[f"0.{nk}"].shape == v.shape:
            clean_dict[f"0.{nk}"] = v
    missing, unexpected = model.load_state_dict(clean_dict, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Matched params: {len(clean_dict)}, missing: {len(missing)}, unexpected: {len(unexpected)}")


def extract_split(args, split, model, device):
    ds = CamelyonPatchImageDataset(args.data_dir, split=split, patch_label_file=args.patch_label_file)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    features = []
    patch_labels = []
    patch_has_gt = []
    slide_labels = []
    slide_indices = []
    slide_names = []

    model.eval()
    with torch.no_grad():
        for images, p_lb, p_has_gt, s_lb, s_idx, s_name in tqdm(loader, desc=f"Extract {split}"):
            images = images.to(device, non_blocking=True)
            feat = model(images).flatten(1).cpu().numpy().astype(np.float32)  # [B,2048]

            features.append(feat)
            patch_labels.append(p_lb.numpy().astype(np.int64))
            patch_has_gt.append(p_has_gt.numpy().astype(np.int64))
            slide_labels.append(s_lb.numpy().astype(np.int64))
            slide_indices.append(s_idx.numpy().astype(np.int64))
            slide_names.extend(list(s_name))

    features = np.concatenate(features, axis=0)
    patch_labels = np.concatenate(patch_labels, axis=0)
    patch_has_gt = np.concatenate(patch_has_gt, axis=0)
    slide_labels = np.concatenate(slide_labels, axis=0)
    slide_indices = np.concatenate(slide_indices, axis=0)
    slide_names = np.array(slide_names).astype(str)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{split}_features.npz")
    np.savez(
        out_path,
        features=features,
        patch_labels=patch_labels,
        patch_has_gt=patch_has_gt,
        slide_labels=slide_labels,
        slide_indices=slide_indices,
        slide_names=slide_names,
    )
    print(f"[Saved] {out_path}")
    print(f"[Saved] features shape={features.shape}")


def get_args():
    p = argparse.ArgumentParser(description="Extract CAMELYON16 patch features with ResNet50")
    p.add_argument("--data_dir", type=str, required=True, help="Root dir that contains training/ and testing/")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to save *features.npz")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--pretrained", action="store_true", default=True, help="Use ImageNet pretrained ResNet50")
    p.add_argument("--model_ckpt", type=str, default="", help="Optional checkpoint to initialize feature extractor")
    p.add_argument("--patch_label_file", type=str, default="", help="Optional patch labels csv/tsv")
    p.add_argument("--split", type=str, default="both", choices=["training", "testing", "both"])
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_resnet50_feature_extractor(pretrained=args.pretrained).to(device)
    if args.model_ckpt:
        try_load_checkpoint(model, args.model_ckpt)

    if args.split in ["training", "both"]:
        extract_split(args, "training", model, device)
    if args.split in ["testing", "both"]:
        extract_split(args, "testing", model, device)


if __name__ == "__main__":
    main()
