"""
CAMELYON16 Slide-aware Dataset
支持两种模式：
1. Instance mode: 返回单个patch（用于SSL和student训练）
2. Bag mode: 返回整个slide的所有patches（用于teacher训练）
"""

import os
import glob
import random
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from semilearn.datasets.augmentation import RandAugment

# ===== 参数配置 =====
DEFAULT_NEGATIVE_POSITIVE_RATIO = 3


class SlideSampler(Sampler):
    """
    自定义采样器：每次采样一个完整的slide
    """

    def __init__(self, slide_indices, shuffle=True):
        """
        Args:
            slide_indices: list of slide indices
            shuffle: whether to shuffle slides
        """
        self.slide_indices = slide_indices
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.slide_indices)).tolist()
        else:
            indices = list(range(len(self.slide_indices)))

        for idx in indices:
            yield self.slide_indices[idx]

    def __len__(self):
        return len(self.slide_indices)


class CAMELYONSlideDataset(Dataset):
    """
    Slide-aware dataset for CAMELYON16

    支持两种返回模式：
    - return_bag=False: 返回单个patch（用于instance-level训练）
    - return_bag=True: 返回整个slide的patches（用于bag-level训练）
    """

    def __init__(self, data_dir, train=True,
                 transform_weak=None, transform_strong=None,
                 return_bag=False, max_patches_per_slide=None,
                 labeled_slides=None, is_labeled=True):
        """
        Args:
            data_dir: 数据根目录
            train: 是否训练集
            transform_weak: 弱数据增强
            transform_strong: 强数据增强
            return_bag: 是否返回整个bag
            max_patches_per_slide: 每个slide最多采样的patch数量
            labeled_slides: 有标注的slide名称列表
            is_labeled: 当前数据集是否为有标注数据
        """
        self.data_dir = data_dir
        self.train = train
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong
        self.return_bag = return_bag
        self.max_patches_per_slide = max_patches_per_slide
        self.labeled_slides = labeled_slides
        self.is_labeled = is_labeled

        # 加载数据
        self._load_data()

    def _load_data(self):
        """加载并组织数据"""
        subset = 'training' if self.train else 'testing'
        root_dir = os.path.join(self.data_dir, subset)

        # 获取所有slide目录
        all_slides = [d for d in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, d))]

        # 如果指定了labeled_slides，只使用这些slides
        if self.labeled_slides is not None:
            all_slides = [s for s in all_slides if s in self.labeled_slides]

        print(f"Loading {subset} data: {len(all_slides)} slides")

        # 组织slide和patch信息
        self.slides = []  # slide名称列表
        self.slide_to_patches = {}  # slide -> patch路径列表
        self.slide_labels = {}  # slide -> label
        self.patch_list = []  # 所有patch的路径
        self.patch_labels = []  # 所有patch的标签
        self.patch_to_slide = []  # patch -> slide_idx

        for slide_idx, slide_name in enumerate(all_slides):
            slide_dir = os.path.join(root_dir, slide_name)

            # 确定slide标签
            slide_label = self._determine_slide_label(slide_name)

            # 获取该slide的所有patches
            image_extensions = ['*.jpg', '*.jpeg', '*.png']
            patch_paths = []
            for ext in image_extensions:
                patch_paths.extend(glob.glob(os.path.join(slide_dir, ext)))

            # 限制每个slide的patch数量
            if self.max_patches_per_slide and len(patch_paths) > self.max_patches_per_slide:
                random.shuffle(patch_paths)
                patch_paths = patch_paths[:self.max_patches_per_slide]

            if len(patch_paths) == 0:
                continue

            # 存储slide信息
            self.slides.append(slide_name)
            self.slide_to_patches[slide_name] = patch_paths
            self.slide_labels[slide_name] = slide_label

            # 存储patch信息
            for patch_path in patch_paths:
                filename = os.path.basename(patch_path)
                patch_label = self._determine_patch_label(filename, slide_label, self.train)

                self.patch_list.append(patch_path)
                self.patch_labels.append(patch_label)
                self.patch_to_slide.append(slide_idx)

        print(f"Loaded {len(self.slides)} slides, {len(self.patch_list)} patches")
        print(f"  Positive patches: {sum(1 for l in self.patch_labels if l == 1)}")
        print(f"  Negative patches: {sum(1 for l in self.patch_labels if l == 0)}")
        print(f"  Unlabeled patches: {sum(1 for l in self.patch_labels if l == -1)}")

    def _determine_slide_label(self, slide_name):
        """确定slide标签"""
        if slide_name.endswith('_0'):
            return 0
        elif slide_name.endswith('_1'):
            return 1
        elif 'normal' in slide_name.lower():
            return 0
        elif 'tumor' in slide_name.lower():
            return 1
        else:
            return 0

    def _determine_patch_label(self, filename, slide_label, is_train):
        """确定patch标签"""
        base_name = os.path.splitext(filename)[0]

        if base_name.endswith('_1'):
            return 1
        elif base_name.endswith('_0'):
            return 0
        else:
            # 无明确标注的patch
            if slide_label == 0:
                return 0  # 阴性slide中的patch为阴性
            else:
                # 阳性slide中的无标注patch
                if is_train and not self.is_labeled:
                    return -1  # 训练时作为无标注
                else:
                    return 1  # 测试时假设为阳性

    def __len__(self):
        if self.return_bag:
            return len(self.slides)
        else:
            return len(self.patch_list)

    def __getitem__(self, idx):
        if self.return_bag:
            return self._get_bag(idx)
        else:
            return self._get_patch(idx)

    def _get_patch(self, idx):
        """返回单个patch（instance mode）"""
        patch_path = self.patch_list[idx]
        patch_label = self.patch_labels[idx]
        slide_idx = self.patch_to_slide[idx]
        slide_name = self.slides[slide_idx]

        # 加载图像
        try:
            image = Image.open(patch_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {patch_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # 数据增强
        if self.transform_weak:
            img_weak = self.transform_weak(image)
        else:
            img_weak = transforms.ToTensor()(image)

        if self.transform_strong:
            img_strong = self.transform_strong(image)
        else:
            img_strong = img_weak

        # 返回格式
        if self.is_labeled:
            return {
                'idx_lb': idx,
                'x_lb': img_weak,
                'y_lb': patch_label,
                'slide_idx': slide_idx,
                'slide_name': slide_name
            }
        else:
            return {
                'idx_ulb': idx,
                'x_ulb_w': img_weak,
                'x_ulb_s': img_strong,
                'y_ulb': patch_label,  # 可能是-1（无标注）
                'slide_idx': slide_idx,
                'slide_name': slide_name
            }

    def _get_bag(self, idx):
        """返回整个slide的patches（bag mode）"""
        slide_name = self.slides[idx]
        patch_paths = self.slide_to_patches[slide_name]
        slide_label = self.slide_labels[slide_name]

        # 加载所有patches
        bag_images = []
        patch_labels = []

        for patch_path in patch_paths:
            try:
                image = Image.open(patch_path).convert('RGB')
            except Exception as e:
                print(f"Error loading {patch_path}: {e}")
                image = Image.new('RGB', (224, 224), (0, 0, 0))

            # 应用变换
            if self.transform_weak:
                image = self.transform_weak(image)
            else:
                image = transforms.ToTensor()(image)

            bag_images.append(image)

            # Patch标签
            filename = os.path.basename(patch_path)
            patch_label = self._determine_patch_label(filename, slide_label, self.train)
            patch_labels.append(patch_label)

        # Stack成tensor
        bag_tensor = torch.stack(bag_images)  # [N, C, H, W]
        patch_labels_tensor = torch.tensor(patch_labels)

        return {
            'bag': bag_tensor,
            'patch_labels': patch_labels_tensor,
            'slide_label': slide_label,
            'slide_idx': idx,
            'slide_name': slide_name
        }

    def get_slide_dataloader(self, batch_size=1, shuffle=True, num_workers=4):
        """
        创建slide-level的dataloader
        每次返回一个完整的slide
        """
        assert self.return_bag, "Must set return_bag=True for slide dataloader"

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_bag
        )

    @staticmethod
    def _collate_bag(batch):
        """
        自定义collate函数：处理不同大小的bags
        由于每个slide的patch数量不同，不能直接stack
        """
        # batch size应该是1（每次一个slide）
        assert len(batch) == 1, "Bag mode only supports batch_size=1"
        return batch[0]


def get_camelyon16_slide_aware(args, alg='instant_dual_slide', dataset='camelyon16',
                               num_labels=500, num_classes=2, data_dir='./data',
                               include_lb_to_ulb=True):
    """
    创建slide-aware的CAMELYON16数据集

    Returns:
        lb_dataset_instance: 有标注数据的instance dataset
        ulb_dataset_instance: 无标注数据的instance dataset
        eval_dataset_instance: 测试集的instance dataset
        lb_dataset_bag: 有标注数据的bag dataset（用于teacher训练）
        eval_dataset_bag: 测试集的bag dataset（用于slide-level评估）
    """

    # 获取参数
    if hasattr(args, 'data_dir'):
        data_dir = args.data_dir

    data_seed = getattr(args, 'data_seed', 42)
    max_patches = getattr(args, 'max_patches_per_slide', 500)
    crop_size = getattr(args, 'img_size', 224)
    crop_ratio = getattr(args, 'crop_ratio', 0.875)

    print(f"Creating slide-aware CAMELYON16 dataset:")
    print(f"  Data dir: {data_dir}")
    print(f"  Max patches per slide: {max_patches}")

    # 数据增强
    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 第一步：选择labeled slides
    # 简化版：加载所有训练slides，随机选择
    training_dir = os.path.join(data_dir, 'training')
    all_slides = [d for d in os.listdir(training_dir)
                  if os.path.isdir(os.path.join(training_dir, d))]

    # 分类slides
    pos_slides = [s for s in all_slides if s.endswith('_1') or 'tumor' in s.lower()]
    neg_slides = [s for s in all_slides if s.endswith('_0') or 'normal' in s.lower()]

    print(f"Available: {len(pos_slides)} positive slides, {len(neg_slides)} negative slides")

    # 随机选择slides
    random.seed(data_seed)
    num_pos = getattr(args, 'num_positive_slides', 10)
    num_neg = getattr(args, 'num_negative_slides', 10)

    selected_pos_slides = random.sample(pos_slides, min(num_pos, len(pos_slides)))
    selected_neg_slides = random.sample(neg_slides, min(num_neg, len(neg_slides)))
    labeled_slides = selected_pos_slides + selected_neg_slides

    print(f"Selected {len(labeled_slides)} slides for labeled data:")
    print(f"  Positive: {selected_pos_slides}")
    print(f"  Negative: {selected_neg_slides}")

    # 创建datasets
    # 1. Labeled instance dataset
    lb_dataset_instance = CAMELYONSlideDataset(
        data_dir, train=True,
        transform_weak=transform_weak,
        return_bag=False,
        max_patches_per_slide=max_patches,
        labeled_slides=labeled_slides,
        is_labeled=True
    )

    # 2. Labeled bag dataset (用于teacher训练)
    lb_dataset_bag = CAMELYONSlideDataset(
        data_dir, train=True,
        transform_weak=transform_weak,
        return_bag=True,
        max_patches_per_slide=max_patches,
        labeled_slides=labeled_slides,
        is_labeled=True
    )

    # 3. Unlabeled instance dataset
    if include_lb_to_ulb:
        # 包含labeled slides
        ulb_slides = all_slides
    else:
        # 排除labeled slides
        ulb_slides = [s for s in all_slides if s not in labeled_slides]

    ulb_dataset_instance = CAMELYONSlideDataset(
        data_dir, train=True,
        transform_weak=transform_weak,
        transform_strong=transform_strong,
        return_bag=False,
        max_patches_per_slide=max_patches,
        labeled_slides=ulb_slides if not include_lb_to_ulb else None,
        is_labeled=False
    )

    # 4. Eval instance dataset
    eval_dataset_instance = CAMELYONSlideDataset(
        data_dir, train=False,
        transform_weak=transform_val,
        return_bag=False,
        max_patches_per_slide=max_patches,
        is_labeled=True
    )

    # 5. Eval bag dataset (用于slide-level评估)
    eval_dataset_bag = CAMELYONSlideDataset(
        data_dir, train=False,
        transform_weak=transform_val,
        return_bag=True,
        max_patches_per_slide=max_patches,
        is_labeled=True
    )

    print("\nDataset summary:")
    print(f"  Labeled instances: {len(lb_dataset_instance)}")
    print(f"  Labeled bags: {len(lb_dataset_bag)}")
    print(f"  Unlabeled instances: {len(ulb_dataset_instance)}")
    print(f"  Eval instances: {len(eval_dataset_instance)}")
    print(f"  Eval bags: {len(eval_dataset_bag)}")

    # 返回instance dataset（用于标准USB接口）
    # 以及额外的bag datasets
    return (lb_dataset_instance, ulb_dataset_instance, eval_dataset_instance,
            lb_dataset_bag, eval_dataset_bag)


if __name__ == '__main__':
    # 测试代码
    class Args:
        data_dir = '/path/to/CAMELYON16'
        data_seed = 42
        max_patches_per_slide = 100
        img_size = 224
        crop_ratio = 0.875
        num_positive_slides = 10
        num_negative_slides = 10


    args = Args()
    datasets = get_camelyon16_slide_aware(args)

    lb_inst, ulb_inst, eval_inst, lb_bag, eval_bag = datasets

    print("\nTesting instance dataset:")
    sample = lb_inst[0]
    print(f"  Keys: {sample.keys()}")
    print(f"  Image shape: {sample['x_lb'].shape}")

    print("\nTesting bag dataset:")
    bag_sample = lb_bag[0]
    print(f"  Keys: {bag_sample.keys()}")
    print(f"  Bag shape: {bag_sample['bag'].shape}")
    print(f"  Num patches: {bag_sample['bag'].shape[0]}")