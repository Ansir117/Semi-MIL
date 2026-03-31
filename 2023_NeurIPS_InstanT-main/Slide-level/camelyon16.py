# camelyon16.py - 修正版，添加slide聚合支持
# 适配真实的CAMELYON16数据格式：slide名称末尾_0/_1表示slide标签，patch名称末尾_0/_1表示patch标签

import os
import glob
import random
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation

# ===== 数据比例配置 (统一修改入口) =====
DEFAULT_NEGATIVE_POSITIVE_RATIO = 3  # 阴性:阳性比例，统一修改此处即可


class PathBasedDataset(Dataset):
    """基于路径的数据集，返回USB框架期望的字典格式，支持slide信息追踪"""

    def __init__(self, paths, targets, alg, num_classes, transform_weak, is_ulb=False, transform_strong=None):
        self.paths = paths
        self.targets = targets
        self.alg = alg
        self.num_classes = num_classes
        self.transform_weak = transform_weak
        self.is_ulb = is_ulb
        self.transform_strong = transform_strong

        # 提取slide信息用于后续聚合
        self.slide_info = self._extract_slide_info()

    def _extract_slide_info(self):
        """提取每个patch对应的slide信息"""
        slide_info = []
        for path in self.paths:
            slide_name = os.path.basename(os.path.dirname(path))
            slide_info.append(slide_name)
        return slide_info

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        target = self.targets[idx]

        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图像 {path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.is_ulb:
            # 无标签数据：返回弱增强和强增强
            if self.transform_weak:
                img_weak = self.transform_weak(image)
            else:
                img_weak = transforms.ToTensor()(image)

            if self.transform_strong:
                img_strong = self.transform_strong(image)
            else:
                img_strong = img_weak

            return {
                'idx_ulb': idx,
                'x_ulb_w': img_weak,
                'x_ulb_s': img_strong,
                'y_ulb': target,
                'path': path,  # 添加路径信息
                'slide_name': self.slide_info[idx]  # 添加slide信息
            }
        else:
            # 有标签数据：只应用弱增强
            if self.transform_weak:
                img = self.transform_weak(image)
            else:
                img = transforms.ToTensor()(image)

            return {
                'idx_lb': idx,
                'x_lb': img,
                'y_lb': target,
                'path': path,  # 添加路径信息
                'slide_name': self.slide_info[idx]  # 添加slide信息
            }


class EvalDatasetWithSlideInfo(Dataset):
    """专门用于评估的数据集，确保返回路径信息用于slide聚合"""

    def __init__(self, paths, targets, transform, slide_info=None):
        self.paths = paths
        self.targets = targets
        self.transform = transform

        if slide_info is None:
            self.slide_info = [os.path.basename(os.path.dirname(path)) for path in paths]
        else:
            self.slide_info = slide_info

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        target = self.targets[idx]
        slide_name = self.slide_info[idx]

        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图像 {path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # 返回元组格式：(image, label, path, slide_name)
        return image, target, path, slide_name


def determine_slide_label(slide_name):
    """
    根据slide名称确定slide标签
    Args:
        slide_name: slide名称，如 'normal_157_0', 'tumor_001_1', 'test_002_1'
    Returns:
        int: 0表示阴性slide，1表示阳性slide
    """
    if slide_name.endswith('_0'):
        return 0  # 阴性slide
    elif slide_name.endswith('_1'):
        return 1  # 阳性slide
    else:
        # 如果没有明确的_0/_1后缀，根据前缀判断
        if slide_name.startswith('normal'):
            return 0
        elif slide_name.startswith('tumor'):
            return 1
        else:
            return 0  # 默认阴性


def determine_patch_label(patch_filename, slide_label):
    """
    根据patch文件名和所属slide确定patch标签
    Args:
        patch_filename: patch文件名，如 'x_y_1.jpg', 'x_y_0.jpg', 'x_y.jpg'
        slide_label: 所属slide的标签 (0=阴性slide, 1=阳性slide)
    Returns:
        int: 0=阴性patch, 1=阳性patch, -1=无标注patch
    """
    # 移除文件扩展名
    base_name = os.path.splitext(patch_filename)[0]

    if base_name.endswith('_1'):
        return 1  # 明确的阳性patch
    elif base_name.endswith('_0'):
        return 0  # 明确的阴性patch
    else:
        # 无明确后缀的patch
        if slide_label == 0:
            # 阴性slide中的无后缀patch为阴性
            return 0
        else:
            # 阳性slide中的无后缀patch为无标注（用于无监督学习）
            return -1


def load_camelyon16_data(data_dir, max_samples_per_slide=None):
    """
    加载CAMELYON16数据，根据真实的数据格式

    Returns:
        tuple: (train_paths, train_labels, train_slide_info, test_paths, test_labels)
    """

    print(f"加载CAMELYON16数据集...")
    print(f"数据目录: {data_dir}")

    training_dir = os.path.join(data_dir, 'training')
    testing_dir = os.path.join(data_dir, 'testing')

    if not os.path.exists(training_dir):
        raise ValueError(f"找不到training目录: {training_dir}")
    if not os.path.exists(testing_dir):
        raise ValueError(f"找不到testing目录: {testing_dir}")

    # 获取所有slide
    training_slides = [d for d in os.listdir(training_dir)
                       if os.path.isdir(os.path.join(training_dir, d))]
    testing_slides = [d for d in os.listdir(testing_dir)
                      if os.path.isdir(os.path.join(testing_dir, d))]

    print(f"Training slides: {len(training_slides)} 个")
    print(f"Testing slides: {len(testing_slides)} 个")

    def process_slide(slide_dir, slide_name, is_test=False):
        """处理单个slide"""
        slide_label = determine_slide_label(slide_name)

        # 查找图像文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(slide_dir, ext)))

        if len(image_files) == 0:
            print(f"警告: {slide_name} 中没有找到图像文件")
            return [], [], {}

        # 限制样本数量
        if max_samples_per_slide:
            image_files = image_files[:max_samples_per_slide]

        paths = []
        labels = []
        patch_stats = {'positive': 0, 'negative': 0, 'unlabeled': 0}

        for img_path in image_files:
            if not os.path.exists(img_path):
                continue

            filename = os.path.basename(img_path)

            if is_test:
                # 测试数据：根据patch文件名确定ground truth
                base_name = os.path.splitext(filename)[0]
                if base_name.endswith('_1'):
                    label = 1  # 阳性
                elif base_name.endswith('_0'):
                    label = 0  # 阴性
                else:
                    label = 0  # 无后缀默认阴性
            else:
                # 训练数据：根据patch文件名和slide标签确定
                label = determine_patch_label(filename, slide_label)

            paths.append(img_path)
            labels.append(label)

            # 统计
            if label == 1:
                patch_stats['positive'] += 1
            elif label == 0:
                patch_stats['negative'] += 1
            else:
                patch_stats['unlabeled'] += 1

        print(f"  {slide_name} (slide_label={slide_label}): {len(paths)} patches, " +
              f"阳性={patch_stats['positive']}, 阴性={patch_stats['negative']}, 无标注={patch_stats['unlabeled']}")

        return paths, labels, {
            'slide_name': slide_name,
            'slide_label': slide_label,
            'patch_count': len(paths),
            'patch_stats': patch_stats
        }

    # 加载训练数据
    train_paths = []
    train_labels = []
    train_slide_info = []

    print("处理训练数据...")
    for slide_name in training_slides:
        slide_dir = os.path.join(training_dir, slide_name)
        paths, labels, info = process_slide(slide_dir, slide_name, is_test=False)

        train_paths.extend(paths)
        train_labels.extend(labels)
        train_slide_info.append(info)

    # 加载测试数据
    test_paths = []
    test_labels = []

    print("处理测试数据...")
    for slide_name in testing_slides:
        slide_dir = os.path.join(testing_dir, slide_name)
        paths, labels, info = process_slide(slide_dir, slide_name, is_test=True)

        test_paths.extend(paths)
        test_labels.extend(labels)

    print(f"\n数据加载完成:")
    print(f"训练集: {len(train_paths)} patches")
    print(f"  阳性: {train_labels.count(1)}")
    print(f"  阴性: {train_labels.count(0)}")
    print(f"  无标注: {train_labels.count(-1)}")
    print(f"测试集: {len(test_paths)} patches")
    print(f"  阳性: {test_labels.count(1)}")
    print(f"  阴性: {test_labels.count(0)}")

    return train_paths, train_labels, train_slide_info, test_paths, test_labels


def select_slides_and_patches_for_ssl(train_paths, train_labels, train_slide_info,
                                      num_positive_slides=10, num_negative_slides=10,
                                      positive_negative_ratio=DEFAULT_NEGATIVE_POSITIVE_RATIO, data_seed=42):
    """
    为半监督学习选择slide和patch

    Args:
        train_paths: 训练数据路径列表
        train_labels: 训练数据标签列表
        train_slide_info: slide信息列表
        num_positive_slides: 选择的阳性slide数量
        num_negative_slides: 选择的阴性slide数量
        positive_negative_ratio: 阴性:阳性比例
        data_seed: 随机种子

    Returns:
        tuple: (labeled_indices, unlabeled_indices, selected_slide_info)
    """

    print(f"半监督学习数据选择 (seed={data_seed}):")
    print(f"目标: 选择 {num_positive_slides} 个阳性slide 和 {num_negative_slides} 个阴性slide")
    print(f"比例: 阴性:阳性 = {positive_negative_ratio}:1")

    random.seed(data_seed)

    # 分类slide
    positive_slides = [info for info in train_slide_info if info['slide_label'] == 1]
    negative_slides = [info for info in train_slide_info if info['slide_label'] == 0]

    print(f"可用阳性slides: {len(positive_slides)} 个")
    print(f"可用阴性slides: {len(negative_slides)} 个")

    # 随机选择slides
    selected_positive_slides = random.sample(positive_slides,
                                             min(num_positive_slides, len(positive_slides)))
    selected_negative_slides = random.sample(negative_slides,
                                             min(num_negative_slides, len(negative_slides)))

    print(f"选中阳性slides: {[s['slide_name'] for s in selected_positive_slides]}")
    print(f"选中阴性slides: {[s['slide_name'] for s in selected_negative_slides]}")

    # 构建路径到slide的映射
    path_to_slide = {}
    for i, path in enumerate(train_paths):
        slide_name = os.path.basename(os.path.dirname(path))
        path_to_slide[i] = slide_name

    # 收集选中slide的有标注数据
    labeled_positive_indices = []
    labeled_negative_candidates = []

    for i, (path, label) in enumerate(zip(train_paths, train_labels)):
        slide_name = path_to_slide[i]

        # 阳性数据：从选中的阳性slide中选择所有阳性patch
        if slide_name in [s['slide_name'] for s in selected_positive_slides] and label == 1:
            labeled_positive_indices.append(i)

        # 阴性候选：从选中的阴性slide中选择所有阴性patch
        elif slide_name in [s['slide_name'] for s in selected_negative_slides] and label == 0:
            labeled_negative_candidates.append(i)

    # 根据比例选择阴性数据
    target_negative_count = len(labeled_positive_indices) * positive_negative_ratio

    if len(labeled_negative_candidates) >= target_negative_count:
        random.seed(data_seed + 1)
        labeled_negative_indices = random.sample(labeled_negative_candidates, target_negative_count)
    else:
        labeled_negative_indices = labeled_negative_candidates
        print(f"警告: 阴性候选数({len(labeled_negative_candidates)})少于目标数({target_negative_count})")

    # 合并有标注数据索引
    labeled_indices = labeled_positive_indices + labeled_negative_indices

    # 其余所有数据作为无标注数据
    unlabeled_indices = [i for i in range(len(train_paths)) if i not in labeled_indices]

    print(f"\n选择结果:")
    print(f"  有标注数据: {len(labeled_indices)} patches")
    print(f"    阳性: {len(labeled_positive_indices)}")
    print(f"    阴性: {len(labeled_negative_indices)}")
    print(f"    实际比例: {len(labeled_negative_indices) / len(labeled_positive_indices):.1f}:1")
    print(f"  无标注数据: {len(unlabeled_indices)} patches")

    selected_slide_info = {
        'positive_slides': [s['slide_name'] for s in selected_positive_slides],
        'negative_slides': [s['slide_name'] for s in selected_negative_slides],
        'positive_patch_count': len(labeled_positive_indices),
        'negative_patch_count': len(labeled_negative_indices)
    }

    return labeled_indices, unlabeled_indices, selected_slide_info


def get_camelyon16(args, alg='instant', dataset='camelyon16', num_labels=500, num_classes=2,
                   data_dir='./data', include_lb_to_ulb=True):
    """
    获取CAMELYON16数据集的SSL划分 - 适配真实数据格式，支持slide级别聚合
    """

    # 获取参数
    if hasattr(args, 'data_dir'):
        data_dir = args.data_dir
    if hasattr(args, 'include_lb_to_ulb'):
        include_lb_to_ulb = args.include_lb_to_ulb

    data_seed = getattr(args, 'data_seed', 42)
    experiment_id = getattr(args, 'experiment_id', -1)
    max_samples = getattr(args, 'max_samples_per_slide', None)

    print(f"初始化CAMELYON16数据集:")
    print(f"  数据目录: {data_dir}")
    print(f"  数据种子: {data_seed}")
    print(f"  实验ID: {experiment_id}")
    print(f"  每slide最大样本: {max_samples}")

    # 数据增强
    crop_size = getattr(args, 'img_size', 224)
    crop_ratio = getattr(args, 'crop_ratio', 0.875)

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据
    train_paths, train_labels, train_slide_info, test_paths, test_labels = load_camelyon16_data(
        data_dir, max_samples
    )

    # 检查数据质量
    if train_labels.count(1) == 0:
        raise ValueError("训练集中没有阳性样本！")
    if train_labels.count(0) == 0:
        raise ValueError("训练集中没有阴性样本！")

    # 选择半监督学习数据
    num_positive_slides = getattr(args, 'num_positive_slides', 10)
    num_negative_slides = getattr(args, 'num_negative_slides', 10)
    positive_negative_ratio = getattr(args, 'negative_positive_ratio', DEFAULT_NEGATIVE_POSITIVE_RATIO)

    labeled_indices, unlabeled_indices, selected_slide_info = select_slides_and_patches_for_ssl(
        train_paths, train_labels, train_slide_info,
        num_positive_slides, num_negative_slides, positive_negative_ratio, data_seed
    )

    # 构建最终数据集
    # 有标注数据
    lb_paths = [train_paths[i] for i in labeled_indices]
    lb_targets = [train_labels[i] for i in labeled_indices]

    # 无标注数据（隐藏真实标签）
    ulb_paths = [train_paths[i] for i in unlabeled_indices]
    ulb_targets = [-1] * len(ulb_paths)  # 隐藏标签

    # 如果include_lb_to_ulb=True，将有标注数据也加入无标注集
    if include_lb_to_ulb:
        for i in labeled_indices:
            ulb_paths.append(train_paths[i])
            ulb_targets.append(train_labels[i])  # 保持真实标签用于一致性正则化

    # 更新args
    args.num_labels = len(lb_paths)

    print(f"\n最终数据划分:")
    print(f"  有标注集: {len(lb_paths)} 样本")
    print(f"    阳性: {lb_targets.count(1)} ({lb_targets.count(1) / len(lb_targets) * 100:.1f}%)")
    print(f"    阴性: {lb_targets.count(0)} ({lb_targets.count(0) / len(lb_targets) * 100:.1f}%)")
    print(f"  无标注集: {len(ulb_paths)} 样本")
    print(f"  测试集: {len(test_paths)} 样本")
    print(f"    阳性: {test_labels.count(1)} ({test_labels.count(1) / len(test_labels) * 100:.1f}%)")
    print(f"    阴性: {test_labels.count(0)} ({test_labels.count(0) / len(test_labels) * 100:.1f}%)")
    print(f"  选中slides: 阳性={selected_slide_info['positive_slides']}")
    print(f"               阴性={selected_slide_info['negative_slides']}")

    # 创建数据集对象
    lb_dataset = PathBasedDataset(
        lb_paths, lb_targets, alg, num_classes, transform_weak,
        is_ulb=False, transform_strong=None
    )

    ulb_dataset = PathBasedDataset(
        ulb_paths, ulb_targets, alg, num_classes, transform_weak,
        is_ulb=True, transform_strong=transform_strong
    )

    # 创建专门的评估数据集，确保包含slide信息
    eval_dataset = EvalDatasetWithSlideInfo(
        test_paths, test_labels, transform_val
    )

    # 为了兼容性，也创建标准格式的评估数据集
    eval_dataset_standard = PathBasedDataset(
        test_paths, test_labels, alg, num_classes, transform_val,
        is_ulb=False, transform_strong=None
    )

    print("CAMELYON16数据集创建成功！")
    print("✅ 支持slide级别聚合评估")

    return lb_dataset, ulb_dataset, eval_dataset