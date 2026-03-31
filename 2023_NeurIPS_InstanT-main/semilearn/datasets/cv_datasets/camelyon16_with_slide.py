"""
Slide-aware CAMELYON16 Dataset
支持slide信息追踪和slide-aware采样
放置位置: semilearn/datasets/cv_datasets/camelyon16_with_slide.py
"""

import os
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


class CAMELYON16WithSlideInfo(Dataset):
    """
    CAMELYON16数据集，支持slide信息追踪
    每个样本返回: (image, {patch_label, slide_label, slide_id, slide_name, path})
    """

    def __init__(
            self,
            data_dir,
            split='train',
            transform=None,
            max_samples_per_slide=None,
            slide_sample_strategy='random'
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.max_samples_per_slide = max_samples_per_slide
        self.slide_sample_strategy = slide_sample_strategy

        if split == 'train':
            self.root_dir = os.path.join(data_dir, 'training')
        else:
            self.root_dir = os.path.join(data_dir, 'testing')

        self.slide_dirs = sorted(glob.glob(os.path.join(self.root_dir, '*')))
        self.patch_list = []
        self.slide_to_patches = {}
        self.slide_names = []
        self.slide_labels = []

        self._build_patch_list()

        print(f"[CAMELYON16] Loaded {len(self.patch_list)} patches from {len(self.slide_dirs)} slides")

    def _build_patch_list(self):
        slide_id = 0

        for slide_dir in self.slide_dirs:
            slide_name = os.path.basename(slide_dir)

            if 'tumor' in slide_name.lower() or '_pos' in slide_name.lower():
                slide_label = 1
            else:
                slide_label = 0

            patch_files = sorted(
                glob.glob(os.path.join(slide_dir, '*.png')) +
                glob.glob(os.path.join(slide_dir, '*.jpg'))
            )

            if len(patch_files) == 0:
                continue

            if self.max_samples_per_slide is not None and len(patch_files) > self.max_samples_per_slide:
                if self.slide_sample_strategy == 'random':
                    np.random.seed(42)
                    indices = np.random.choice(len(patch_files), self.max_samples_per_slide, replace=False)
                    patch_files = [patch_files[i] for i in sorted(indices)]

            start_idx = len(self.patch_list)

            for patch_file in patch_files:
                if '_pos' in os.path.basename(patch_file):
                    patch_label = 1
                else:
                    patch_label = slide_label

                self.patch_list.append((patch_file, patch_label, slide_id, slide_name))

            end_idx = len(self.patch_list)
            self.slide_to_patches[slide_id] = list(range(start_idx, end_idx))
            self.slide_names.append(slide_name)
            self.slide_labels.append(slide_label)

            slide_id += 1

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        path, patch_label, slide_id, slide_name = self.patch_list[idx]

        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        slide_label = self.slide_labels[slide_id]

        label_dict = {
            'patch_label': patch_label,
            'slide_label': slide_label,
            'slide_id': slide_id,
            'slide_name': slide_name,
            'path': path
        }

        return image, label_dict

    def get_slide_info(self, slide_id):
        return self.slide_to_patches.get(slide_id, [])

    def get_num_slides(self):
        return len(self.slide_names)


class SlideAwareSampler(Sampler):
    """确保每个batch包含来自多个不同slide的patches"""

    def __init__(self, dataset, batch_size, patches_per_slide=4, drop_last=True, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.patches_per_slide = patches_per_slide
        self.drop_last = drop_last
        self.shuffle = shuffle

        self.slides_per_batch = batch_size // patches_per_slide
        if self.slides_per_batch == 0:
            raise ValueError(f"batch_size must be >= patches_per_slide")

        self.num_slides = dataset.get_num_slides()
        self.actual_batch_size = self.slides_per_batch * self.patches_per_slide

    def __iter__(self):
        all_slide_ids = list(range(self.num_slides))

        if self.shuffle:
            np.random.shuffle(all_slide_ids)

        for i in range(0, self.num_slides, self.slides_per_batch):
            batch_slide_ids = all_slide_ids[i:i + self.slides_per_batch]

            if len(batch_slide_ids) < self.slides_per_batch:
                if self.drop_last:
                    break

            batch_indices = []
            for slide_id in batch_slide_ids:
                patch_indices = self.dataset.get_slide_info(slide_id)

                if len(patch_indices) == 0:
                    continue

                if len(patch_indices) >= self.patches_per_slide:
                    sampled = np.random.choice(patch_indices, self.patches_per_slide, replace=False)
                else:
                    sampled = np.random.choice(patch_indices, self.patches_per_slide, replace=True)

                batch_indices.extend(sampled.tolist())

            if self.shuffle:
                np.random.shuffle(batch_indices)

            for idx in batch_indices:
                yield idx

    def __len__(self):
        num_batches = self.num_slides // self.slides_per_batch
        if not self.drop_last and self.num_slides % self.slides_per_batch != 0:
            num_batches += 1
        return num_batches * self.actual_batch_size


def collate_fn_with_slide_info(batch):
    """处理包含slide信息的batch"""
    images = []
    patch_labels = []
    slide_labels = []
    slide_ids = []
    slide_names = []
    paths = []

    for image, label_dict in batch:
        images.append(image)
        patch_labels.append(label_dict['patch_label'])
        slide_labels.append(label_dict['slide_label'])
        slide_ids.append(label_dict['slide_id'])
        slide_names.append(label_dict['slide_name'])
        paths.append(label_dict['path'])

    images = torch.stack(images)
    patch_labels = torch.tensor(patch_labels, dtype=torch.long)
    slide_labels = torch.tensor(slide_labels, dtype=torch.long)
    slide_ids = torch.tensor(slide_ids, dtype=torch.long)

    labels = {
        'patch_label': patch_labels,
        'slide_label': slide_labels,
        'slide_id': slide_ids,
        'slide_name': slide_names,
        'path': paths
    }

    return images, labels