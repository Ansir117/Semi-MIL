# camelyon_ssl.py
# CAMELYON16数据集半监督学习适配器 - 完整版
# 策略2：模拟半监督（推荐）

import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import glob
import random
from collections import defaultdict
from semilearn.datasets.utils import split_ssl_data
from semilearn.datasets.cv_datasets import BasicDataset


def get_camelyon16(args, alg, name, num_labels, num_classes, data_dir=None, include_lb_to_ulb=True):
    """
    获取CAMELYON16数据集，兼容USB框架

    Args:
        args: 参数配置
        alg: 算法名称
        name: 数据集名称
        num_labels: 每类有标签样本数
        num_classes: 类别数量
        data_dir: 数据根目录
        include_lb_to_ulb: 是否将有标签数据包含到无标签数据中
    """
    data_dir = data_dir if data_dir is not None else args.data_dir

    # 定义数据变换
    crop_size = args.img_size

    transform_weak = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.15),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_val = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 加载数据
    ssl_strategy = getattr(args, 'ssl_strategy', 'simulated')  # 默认使用模拟半监督

    if ssl_strategy == 'real':
        # 策略1：真实半监督
        train_data, train_targets, unlabeled_data = load_camelyon_real_ssl(data_dir, args.seed)
        test_data, test_targets = load_camelyon_test(data_dir, args.seed)

        # 真实半监督：直接使用预定义的分割
        lb_data, lb_targets = train_data, train_targets
        ulb_data, ulb_targets = unlabeled_data, [-1] * len(unlabeled_data)  # 无标签用-1表示

    else:
        # 策略2：模拟半监督（推荐）
        all_labeled_data, all_labeled_targets, extra_unlabeled = load_camelyon_simulated_ssl(data_dir, args.seed)

        # 从有标签数据中分割
        lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(
            args, all_labeled_data, all_labeled_targets, num_classes,
            lb_num_labels=num_labels, ulb_num_labels=args.ulb_num_labels,
            lb_imbalance_ratio=args.lb_imb_ratio,
            ulb_imbalance_ratio=args.ulb_imb_ratio,
            include_lb_to_ulb=include_lb_to_ulb
        )

        # 添加真正的无标签数据（额外奖励）
        if len(extra_unlabeled) > 0:
            ulb_data = np.concatenate([ulb_data, extra_unlabeled])
            ulb_targets = np.concatenate([ulb_targets, [-1] * len(extra_unlabeled)])

        # 测试集使用所有有标签数据的一部分
        test_data, test_targets = create_test_set(all_labeled_data, all_labeled_targets, args.seed)

    print(f"=== CAMELYON16 SSL Dataset (Strategy: {ssl_strategy}) ===")
    print(f"Labeled: {len(lb_data)} samples")
    print(f"Unlabeled: {len(ulb_data)} samples")
    print(f"Test: {len(test_data)} samples")

    # 统计标签分布
    if len(lb_targets) > 0:
        unique_lb, counts_lb = np.unique(lb_targets, return_counts=True)
        print(f"Labeled distribution: {dict(zip(unique_lb, counts_lb))}")

    if len(test_targets) > 0:
        unique_test, counts_test = np.unique(test_targets, return_counts=True)
        print(f"Test distribution: {dict(zip(unique_test, counts_test))}")

    # 创建数据集
    lb_dset = CamelyonDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)
    ulb_dset = CamelyonDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)
    eval_dset = CamelyonDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset


def load_camelyon_real_ssl(data_dir, seed=42):
    """
    策略1：真实半监督学习数据加载
    """
    random.seed(seed)
    np.random.seed(seed)

    neg_dir = os.path.join(data_dir, "Neg_Slide")
    pos_dir = os.path.join(data_dir, "Pos_Slide")

    labeled_data = []
    labeled_targets = []
    unlabeled_data = []

    # 处理阴性slide
    neg_slides = [d for d in os.listdir(neg_dir) if os.path.isdir(os.path.join(neg_dir, d))]
    neg_patches_all = []
    for slide in neg_slides:
        slide_dir = os.path.join(neg_dir, slide)
        patches = glob.glob(os.path.join(slide_dir, "*.jpg"))
        neg_patches_all.extend(patches)

    # 从阴性patch中抽取一部分作为有标签数据
    random.shuffle(neg_patches_all)
    neg_labeled_size = int(len(neg_patches_all) * 0.1)  # 10%作为有标签

    labeled_data.extend(neg_patches_all[:neg_labeled_size])
    labeled_targets.extend([0] * neg_labeled_size)
    unlabeled_data.extend(neg_patches_all[neg_labeled_size:])

    # 处理阳性slide
    pos_slides = [d for d in os.listdir(pos_dir) if os.path.isdir(os.path.join(pos_dir, d))]
    for slide in pos_slides:
        slide_dir = os.path.join(pos_dir, slide)
        all_patches = glob.glob(os.path.join(slide_dir, "*.jpg"))

        for patch_path in all_patches:
            patch_name = os.path.basename(patch_path)
            if patch_name.endswith('_1.jpg'):
                labeled_data.append(patch_path)
                labeled_targets.append(1)
            else:
                unlabeled_data.append(patch_path)

    return np.array(labeled_data), np.array(labeled_targets), np.array(unlabeled_data)


def load_camelyon_simulated_ssl(data_dir, seed=42):
    """
    策略2：模拟半监督学习数据加载（推荐）
    """
    random.seed(seed)
    np.random.seed(seed)

    neg_dir = os.path.join(data_dir, "Neg_Slide")
    pos_dir = os.path.join(data_dir, "Pos_Slide")

    print(f"Loading CAMELYON16 data from:")
    print(f"  Negative slides: {neg_dir}")
    print(f"  Positive slides: {pos_dir}")

    all_labeled_data = []
    all_labeled_targets = []
    extra_unlabeled = []

    # 收集所有阴性patch（都有标签）
    if os.path.exists(neg_dir):
        neg_slides = [d for d in os.listdir(neg_dir) if os.path.isdir(os.path.join(neg_dir, d))]
        neg_count = 0
        for slide in neg_slides:
            slide_dir = os.path.join(neg_dir, slide)
            patches = glob.glob(os.path.join(slide_dir, "*.jpg"))
            all_labeled_data.extend(patches)
            all_labeled_targets.extend([0] * len(patches))
            neg_count += len(patches)
        print(f"Loaded {neg_count} negative patches from {len(neg_slides)} slides")

    # 收集标注的阳性patch和未标注patch
    if os.path.exists(pos_dir):
        pos_slides = [d for d in os.listdir(pos_dir) if os.path.isdir(os.path.join(pos_dir, d))]
        pos_labeled_count = 0
        pos_unlabeled_count = 0

        for slide in pos_slides:
            slide_dir = os.path.join(pos_dir, slide)
            all_patches = glob.glob(os.path.join(slide_dir, "*.jpg"))

            for patch_path in all_patches:
                patch_name = os.path.basename(patch_path)
                if patch_name.endswith('_1.jpg'):
                    # 标注的阳性patch
                    all_labeled_data.append(patch_path)
                    all_labeled_targets.append(1)
                    pos_labeled_count += 1
                else:
                    # 未标注的patch（作为额外的无标签数据）
                    extra_unlabeled.append(patch_path)
                    pos_unlabeled_count += 1

        print(f"Loaded {pos_labeled_count} positive patches from {len(pos_slides)} slides")
        print(f"Found {pos_unlabeled_count} extra unlabeled patches")

    print(f"Total labeled data: {len(all_labeled_data)} ({len(all_labeled_targets)} targets)")
    print(f"Extra unlabeled data: {len(extra_unlabeled)}")

    # 统计标签分布
    unique_labels, counts = np.unique(all_labeled_targets, return_counts=True)
    print(f"Label distribution: {dict(zip(unique_labels, counts))}")

    return np.array(all_labeled_data), np.array(all_labeled_targets), np.array(extra_unlabeled)


def create_test_set(all_data, all_targets, seed=42, test_ratio=0.2):
    """
    从所有数据中创建测试集
    """
    random.seed(seed)
    np.random.seed(seed)

    # 按类别分层抽样
    unique_labels = np.unique(all_targets)
    test_data = []
    test_targets = []

    for label in unique_labels:
        label_indices = np.where(all_targets == label)[0]
        n_test = int(len(label_indices) * test_ratio)

        test_indices = np.random.choice(label_indices, n_test, replace=False)
        test_data.extend(all_data[test_indices])
        test_targets.extend([label] * n_test)

    return np.array(test_data), np.array(test_targets)


def load_camelyon_test(data_dir, seed=42):
    """
    加载测试数据（如果有单独的测试集）
    """
    # 如果有单独的测试集目录，在这里加载
    # 现在返回空数组，测试集将从训练数据中分割
    return np.array([]), np.array([])


class CamelyonDataset(BasicDataset):
    """
    CAMELYON16数据集类，继承自BasicDataset
    """

    def __getitem__(self, idx):
        """
        重写getitem方法，处理图像文件路径
        """
        idx = idx % len(self.data)

        # 加载图像（data存储的是文件路径）
        try:
            img = Image.open(self.data[idx]).convert('RGB')
        except Exception as e:
            print(f"Error loading image {self.data[idx]}: {e}")
            # 返回一个dummy图像
            img = Image.new('RGB', (224, 224), color='white')

        target = self.targets[idx]

        # 处理无标签数据（target=-1）
        if target == -1:
            target = 0  # 给无标签数据一个dummy target，在训练时会被忽略

        # 应用变换
        if self.transform is not None:
            img = self.transform(img)

        if not self.is_ulb:
            return img, target
        else:
            if self.strong_transform is not None:
                # 重新加载图像用于强增强
                try:
                    img_strong = self.strong_transform(Image.open(self.data[idx]).convert('RGB'))
                except Exception as e:
                    print(f"Error loading image for strong augmentation {self.data[idx]}: {e}")
                    img_strong = img  # 使用弱增强的图像
                return idx, img, img_strong
            else:
                return idx, img, img


def analyze_camelyon_dataset(data_dir):
    """
    分析CAMELYON16数据集的详细统计信息
    """
    neg_dir = os.path.join(data_dir, "Neg_Slide")
    pos_dir = os.path.join(data_dir, "Pos_Slide")

    print("=== CAMELYON16 Dataset Analysis ===")

    total_neg = 0
    total_pos_labeled = 0
    total_pos_unlabeled = 0

    # 分析阴性数据
    if os.path.exists(neg_dir):
        neg_slides = [d for d in os.listdir(neg_dir) if os.path.isdir(os.path.join(neg_dir, d))]
        print(f"\n📁 Negative Slides ({len(neg_slides)} slides):")
        for slide in neg_slides:
            slide_dir = os.path.join(neg_dir, slide)
            patches = len(glob.glob(os.path.join(slide_dir, "*.jpg")))
            total_neg += patches
            print(f"  {slide}: {patches} patches")
        print(f"Total negative patches: {total_neg}")
    else:
        print(f"❌ Negative directory not found: {neg_dir}")

    # 分析阳性数据
    if os.path.exists(pos_dir):
        pos_slides = [d for d in os.listdir(pos_dir) if os.path.isdir(os.path.join(pos_dir, d))]
        print(f"\n📁 Positive Slides ({len(pos_slides)} slides):")

        for slide in pos_slides:
            slide_dir = os.path.join(pos_dir, slide)
            all_patches = glob.glob(os.path.join(slide_dir, "*.jpg"))
            labeled = sum(1 for p in all_patches if os.path.basename(p).endswith('_1.jpg'))
            unlabeled = len(all_patches) - labeled
            total_pos_labeled += labeled
            total_pos_unlabeled += unlabeled
            print(f"  {slide}: {labeled} labeled, {unlabeled} unlabeled ({len(all_patches)} total)")

        print(f"Total positive labeled patches: {total_pos_labeled}")
        print(f"Total positive unlabeled patches: {total_pos_unlabeled}")
    else:
        print(f"❌ Positive directory not found: {pos_dir}")

    # 总结
    total_labeled = total_neg + total_pos_labeled
    total_unlabeled = total_pos_unlabeled

    print(f"\n📊 Summary:")
    print(f"Total labeled patches: {total_labeled}")
    print(f"  - Negative: {total_neg}")
    print(f"  - Positive: {total_pos_labeled}")
    print(f"Total unlabeled patches: {total_unlabeled}")
    print(f"Grand total: {total_labeled + total_unlabeled}")
    print("===================================")

    return {
        'total_labeled': total_labeled,
        'total_negative': total_neg,
        'total_positive_labeled': total_pos_labeled,
        'total_unlabeled': total_unlabeled
    }


if __name__ == "__main__":
    # 测试数据集加载
    data_dir = "/home/xiaoyuan/Data3/CAMELYON16"
    if os.path.exists(data_dir):
        analyze_camelyon_dataset(data_dir)
    else:
        print(f"Data directory not found: {data_dir}")
        print("Please update the path in this file.")