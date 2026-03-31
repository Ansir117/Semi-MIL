# camelyon16_redesigned.py
# 重新设计的CAMELYON16数据集，遵循官方训练/测试划分和以slide为单位的阳性样本选择

import os
import glob
import random
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation


class PathBasedDataset(Dataset):
    """基于路径的数据集，返回USB框架期望的字典格式"""

    def __init__(self, paths, targets, alg, num_classes, transform_weak, is_ulb=False, transform_strong=None):
        """
        Args:
            paths: 图像路径列表
            targets: 标签列表
            alg: 算法名称
            num_classes: 类别数
            transform_weak: 弱数据增强
            is_ulb: 是否为无标签数据
            transform_strong: 强数据增强（仅用于无标签数据）
        """
        self.paths = paths
        self.targets = targets
        self.alg = alg
        self.num_classes = num_classes
        self.transform_weak = transform_weak
        self.is_ulb = is_ulb
        self.transform_strong = transform_strong

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        target = self.targets[idx]

        # 在使用时才加载图像
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图像 {path}: {e}")
            # 返回空白图像
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
                # 如果没有强增强，使用弱增强
                img_strong = img_weak

            # 返回USB框架期望的无标签数据字典格式
            return {
                'idx_ulb': idx,
                'x_ulb_w': img_weak,
                'x_ulb_s': img_strong,
                'y_ulb': target
            }
        else:
            # 有标签数据：只应用弱增强
            if self.transform_weak:
                img = self.transform_weak(image)
            else:
                img = transforms.ToTensor()(image)

            # 返回USB框架期望的有标签数据字典格式
            return {
                'idx_lb': idx,
                'x_lb': img,
                'y_lb': target
            }


def collect_camelyon16_paths_by_official_split(data_dir, max_samples_per_slide=None):
    """
    自动适配CAMELYON16数据目录结构收集数据路径
    支持两种目录结构：
    1. 新格式：tumor_xxx, normal_xxx, test_xxx 文件夹
    2. 原格式：Neg_Slide/, Pos_Slide/ 文件夹
    """

    print(f"📁 自动检测并收集CAMELYON16数据路径...")

    # 获取所有文件夹
    all_dirs = [d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))]

    print(f"  发现文件夹: {all_dirs}")

    # 检测目录结构类型
    has_new_format = any(d.startswith(('tumor', 'normal', 'test')) for d in all_dirs)
    has_old_format = any(d in ['Neg_Slide', 'Pos_Slide'] for d in all_dirs)

    if has_new_format:
        print("  🔍 检测到新格式目录结构 (tumor_*, normal_*, test_*)")
        return collect_new_format_data(data_dir, all_dirs, max_samples_per_slide)
    elif has_old_format:
        print("  🔍 检测到原格式目录结构 (Neg_Slide/, Pos_Slide/)")
        return collect_old_format_data(data_dir, all_dirs, max_samples_per_slide)
    else:
        raise ValueError(f"无法识别的目录结构！\n发现的文件夹: {all_dirs}\n" +
                         "期望格式1: tumor_*, normal_*, test_*\n" +
                         "期望格式2: Neg_Slide/, Pos_Slide/")


def collect_new_format_data(data_dir, all_dirs, max_samples_per_slide):
    """处理新格式数据 (tumor_*, normal_*, test_*)"""

    # 按照文件夹前缀分类
    train_tumor_dirs = [d for d in all_dirs if d.startswith('tumor')]
    train_normal_dirs = [d for d in all_dirs if d.startswith('normal')]
    test_dirs = [d for d in all_dirs if d.startswith('test')]

    print(f"  训练集肿瘤文件夹: {len(train_tumor_dirs)}")
    print(f"  训练集正常文件夹: {len(train_normal_dirs)}")
    print(f"  测试集文件夹: {len(test_dirs)}")

    # 训练集数据收集
    train_labeled_paths = []
    train_labeled_targets = []
    train_unlabeled_paths = []

    # 测试集数据收集
    test_paths = []
    test_targets = []


def collect_new_format_data(data_dir, all_dirs, max_samples_per_slide):
    """处理新格式数据 (tumor_*, normal_*, test_*)"""

    # 按照文件夹前缀分类，现在支持_0/_1后缀
    train_tumor_dirs = [d for d in all_dirs if d.startswith('tumor')]
    train_normal_dirs = [d for d in all_dirs if d.startswith('normal')]
    test_dirs = [d for d in all_dirs if d.startswith('test')]

    print(f"  训练集肿瘤文件夹: {len(train_tumor_dirs)}")
    print(f"  训练集正常文件夹: {len(train_normal_dirs)}")
    print(f"  测试集文件夹: {len(test_dirs)}")

    # 训练集数据收集
    train_labeled_paths = []
    train_labeled_targets = []
    train_unlabeled_paths = []

    # 测试集数据收集
    test_paths = []
    test_targets = []

    def process_slide_directory_new(slide_base_dir, slide_name, is_test=False):
        """处理单个slide目录 - 新格式，支持_0/_1后缀识别"""
        slide_path = os.path.join(slide_base_dir, slide_name)

        # 根据文件夹命名判断slide类型
        is_positive_slide = slide_name.endswith('_1')
        is_negative_slide = slide_name.endswith('_0')

        print(
            f"    处理slide: {slide_name} ({'阳性slide' if is_positive_slide else '阴性slide' if is_negative_slide else '未知类型'})")

        # 获取图像文件路径
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(slide_path, ext)))

        # 限制样本数量（防止内存问题）
        if max_samples_per_slide:
            image_files = image_files[:max_samples_per_slide]

        labeled_paths_slide = []
        labeled_targets_slide = []
        unlabeled_paths_slide = []

        for img_path in image_files:
            filename = os.path.basename(img_path)

            # 验证文件存在且可读
            if not os.path.exists(img_path):
                continue

            if is_test:
                # 测试集：所有数据都加入测试集，标签根据slide类型确定
                test_paths.append(img_path)
                if is_positive_slide:
                    test_targets.append(1)  # 阳性slide
                elif is_negative_slide:
                    test_targets.append(0)  # 阴性slide
                else:
                    # 如果slide命名不明确，根据文件名判断
                    if filename.endswith('_1.jpg') or filename.endswith('_1.png'):
                        test_targets.append(1)
                    else:
                        test_targets.append(0)
            else:
                # 训练集：优先根据文件名后缀分类，其次根据slide类型
                if filename.endswith('_0.jpg') or filename.endswith('_0.png'):
                    # 明确的阴性标签
                    labeled_paths_slide.append(img_path)
                    labeled_targets_slide.append(0)

                elif filename.endswith('_1.jpg') or filename.endswith('_1.png'):
                    # 明确的阳性标签
                    labeled_paths_slide.append(img_path)
                    labeled_targets_slide.append(1)

                else:
                    # 无明确标签的文件，根据slide类型推断或作为无标签数据
                    if is_positive_slide:
                        # 阳性slide中的无后缀文件，可能是阳性（根据需要调整）
                        unlabeled_paths_slide.append(img_path)
                    elif is_negative_slide:
                        # 阴性slide中的无后缀文件，可能是阴性（根据需要调整）
                        unlabeled_paths_slide.append(img_path)
                    else:
                        # 无法判断的无标签数据
                        unlabeled_paths_slide.append(img_path)

        return labeled_paths_slide, labeled_targets_slide, unlabeled_paths_slide

    # 处理训练集tumor文件夹
    print("  处理训练集tumor文件夹...")
    for slide_name in train_tumor_dirs:
        labeled_paths_slide, labeled_targets_slide, unlabeled_paths_slide = process_slide_directory_new(
            data_dir, slide_name, is_test=False
        )
        train_labeled_paths.extend(labeled_paths_slide)
        train_labeled_targets.extend(labeled_targets_slide)
        train_unlabeled_paths.extend(unlabeled_paths_slide)

    # 处理训练集normal文件夹
    print("  处理训练集normal文件夹...")
    for slide_name in train_normal_dirs:
        labeled_paths_slide, labeled_targets_slide, unlabeled_paths_slide = process_slide_directory_new(
            data_dir, slide_name, is_test=False
        )
        train_labeled_paths.extend(labeled_paths_slide)
        train_labeled_targets.extend(labeled_targets_slide)
        train_unlabeled_paths.extend(unlabeled_paths_slide)

    # 处理测试集
    print("  处理测试集文件夹...")
    for slide_name in test_dirs:
        process_slide_directory_new(data_dir, slide_name, is_test=True)

    print(f"📊 新格式数据收集完成:")
    print(f"  训练集:")
    print(f"    - 有标签路径: {len(train_labeled_paths)}")
    print(f"      * 阴性样本: {train_labeled_targets.count(0)}")
    print(f"      * 阳性样本: {train_labeled_targets.count(1)}")
    print(f"    - 无标签路径: {len(train_unlabeled_paths)}")
    print(f"  测试集:")
    print(f"    - 总路径: {len(test_paths)}")
    print(f"      * 阴性样本: {test_targets.count(0)}")
    print(f"      * 阳性样本: {test_targets.count(1)}")

    return (train_labeled_paths, train_labeled_targets, train_unlabeled_paths,
            test_paths, test_targets)

    print(f"📊 新格式数据收集完成:")
    print(f"  训练集:")
    print(f"    - 有标签路径: {len(train_labeled_paths)}")
    print(f"      * 阴性样本: {train_labeled_targets.count(0)}")
    print(f"      * 阳性样本: {train_labeled_targets.count(1)}")
    print(f"    - 无标签路径: {len(train_unlabeled_paths)}")
    print(f"  测试集:")
    print(f"    - 总路径: {len(test_paths)}")
    print(f"      * 阴性样本: {test_targets.count(0)}")
    print(f"      * 阳性样本: {test_targets.count(1)}")

    return (train_labeled_paths, train_labeled_targets, train_unlabeled_paths,
            test_paths, test_targets)


def collect_old_format_data(data_dir, all_dirs, max_samples_per_slide):
    """处理原格式数据 (Neg_Slide/, Pos_Slide/)，支持新的_0/_1命名规则"""

    # 检查目录结构
    neg_slide_dir = os.path.join(data_dir, 'Neg_Slide')
    pos_slide_dir = os.path.join(data_dir, 'Pos_Slide')

    if not os.path.exists(neg_slide_dir):
        raise ValueError(f"找不到 Neg_Slide 目录: {neg_slide_dir}")
    if not os.path.exists(pos_slide_dir):
        raise ValueError(f"找不到 Pos_Slide 目录: {pos_slide_dir}")

    print(f"  处理原格式目录: Neg_Slide/, Pos_Slide/ (支持_0/_1后缀)")

    train_labeled_paths = []
    train_labeled_targets = []
    train_unlabeled_paths = []

    def process_slide_directory_old(slide_base_dir, slide_type):
        """处理单个slide目录 - 原格式，支持新的_0/_1命名规则"""
        slide_dirs = [d for d in os.listdir(slide_base_dir)
                      if os.path.isdir(os.path.join(slide_base_dir, d))]

        # 按照新的命名规则分类slide
        positive_slides = [d for d in slide_dirs if d.endswith('_1')]
        negative_slides = [d for d in slide_dirs if d.endswith('_0')]
        unknown_slides = [d for d in slide_dirs if not (d.endswith('_0') or d.endswith('_1'))]

        print(f"    处理{slide_type}目录: {len(slide_dirs)}个slide")
        print(f"      阳性slide(_1): {len(positive_slides)}")
        print(f"      阴性slide(_0): {len(negative_slides)}")
        print(f"      未分类slide: {len(unknown_slides)}")

        processed_slides = 0
        total_labeled = 0
        total_unlabeled = 0

        for slide_name in slide_dirs:
            slide_path = os.path.join(slide_base_dir, slide_name)

            # 根据新的命名规则判断slide类型
            is_positive_slide = slide_name.endswith('_1')
            is_negative_slide = slide_name.endswith('_0')

            # 获取图像文件路径
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(slide_path, ext)))

            # 限制样本数量（防止内存问题）
            if max_samples_per_slide:
                image_files = image_files[:max_samples_per_slide]

            labeled_count = 0
            unlabeled_count = 0

            for img_path in image_files:
                filename = os.path.basename(img_path)

                # 验证文件存在且可读
                if not os.path.exists(img_path):
                    continue

                # 根据文件名后缀分类（优先级最高）
                if filename.endswith('_0.jpg') or filename.endswith('_0.png'):
                    # 明确的阴性标签
                    train_labeled_paths.append(img_path)
                    train_labeled_targets.append(0)
                    labeled_count += 1

                elif filename.endswith('_1.jpg') or filename.endswith('_1.png'):
                    # 明确的阳性标签
                    train_labeled_paths.append(img_path)
                    train_labeled_targets.append(1)
                    labeled_count += 1

                else:
                    # 无明确文件名后缀，根据slide命名推断或作为无标签
                    if is_positive_slide:
                        # 阳性slide中的无后缀文件，作为无标签数据
                        train_unlabeled_paths.append(img_path)
                        unlabeled_count += 1
                    elif is_negative_slide:
                        # 阴性slide中的无后缀文件，作为无标签数据
                        train_unlabeled_paths.append(img_path)
                        unlabeled_count += 1
                    else:
                        # 未分类slide中的文件，作为无标签数据
                        train_unlabeled_paths.append(img_path)
                        unlabeled_count += 1

            total_labeled += labeled_count
            total_unlabeled += unlabeled_count
            processed_slides += 1

            # 每处理10个slide显示一次进度
            if processed_slides % 10 == 0:
                print(f"      已处理 {processed_slides}/{len(slide_dirs)} slides")

        print(f"    {slide_type}目录完成: 有标签={total_labeled}, 无标签={total_unlabeled}")

    # 处理阴性和阳性slide
    process_slide_directory_old(neg_slide_dir, "阴性")
    process_slide_directory_old(pos_slide_dir, "阳性")

    # 原格式没有专门的测试集，从训练数据中划分
    # 使用10%的有标签数据作为测试集
    test_size = max(100, len(train_labeled_paths) // 10)

    # 确保测试集中有两个类别的样本
    class_0_indices = [i for i, label in enumerate(train_labeled_targets) if label == 0]
    class_1_indices = [i for i, label in enumerate(train_labeled_targets) if label == 1]

    test_size_per_class = test_size // 2
    test_size_per_class = min(test_size_per_class, len(class_0_indices) // 4, len(class_1_indices) // 4)

    if test_size_per_class > 0:
        random.seed(42)
        test_indices_0 = random.sample(class_0_indices, test_size_per_class)
        test_indices_1 = random.sample(class_1_indices, test_size_per_class)
        test_indices = test_indices_0 + test_indices_1

        test_paths = [train_labeled_paths[i] for i in test_indices]
        test_targets = [train_labeled_targets[i] for i in test_indices]

        # 从训练集中移除测试样本
        train_labeled_paths = [path for i, path in enumerate(train_labeled_paths) if i not in test_indices]
        train_labeled_targets = [target for i, target in enumerate(train_labeled_targets) if i not in test_indices]
    else:
        # 如果样本太少，使用前100个作为测试集
        test_paths = train_labeled_paths[:100]
        test_targets = train_labeled_targets[:100]

    print(f"📊 原格式数据收集完成:")
    print(f"  训练集:")
    print(f"    - 有标签路径: {len(train_labeled_paths)}")
    print(f"      * 阴性样本: {train_labeled_targets.count(0)}")
    print(f"      * 阳性样本: {train_labeled_targets.count(1)}")
    print(f"    - 无标签路径: {len(train_unlabeled_paths)}")
    print(f"  测试集 (从训练集划分):")
    print(f"    - 总路径: {len(test_paths)}")
    print(f"      * 阴性样本: {test_targets.count(0)}")
    print(f"      * 阳性样本: {test_targets.count(1)}")

    return (train_labeled_paths, train_labeled_targets, train_unlabeled_paths,
            test_paths, test_targets)

    print(f"📊 官方划分数据收集完成:")
    print(f"  训练集:")
    print(f"    - 有标签路径: {len(train_labeled_paths)}")
    print(f"      * 阴性样本: {train_labeled_targets.count(0)}")
    print(f"      * 阳性样本: {train_labeled_targets.count(1)}")
    print(f"    - 无标签路径: {len(train_unlabeled_paths)}")
    print(f"  测试集:")
    print(f"    - 总路径: {len(test_paths)}")
    print(f"      * 阴性样本: {test_targets.count(0)}")
    print(f"      * 阳性样本: {test_targets.count(1)}")

    return (train_labeled_paths, train_labeled_targets, train_unlabeled_paths,
            test_paths, test_targets)


def select_positive_samples_by_slide(labeled_paths, labeled_targets, num_slides):
    """
    固定选择指定数量的阳性slide，使用这些slide的所有阳性patch
    现在支持新的_1后缀命名规则来准确识别阳性slide
    """
    print(f"🎯 固定选择 {num_slides} 个阳性slide的所有阳性样本...")

    # 按slide分组阳性样本，优先使用新的命名规则
    slide_to_positive_samples = defaultdict(list)
    slide_type_info = defaultdict(lambda: {'has_positive_patches': False, 'is_positive_slide': False})

    for i, (path, target) in enumerate(zip(labeled_paths, labeled_targets)):
        if target == 1:  # 只处理阳性样本
            # 从路径中提取slide名称（假设路径格式为 .../slide_name/patch.jpg）
            slide_name = os.path.basename(os.path.dirname(path))
            slide_to_positive_samples[slide_name].append(i)
            slide_type_info[slide_name]['has_positive_patches'] = True

            # 检查slide是否以_1结尾（新命名规则）
            if slide_name.endswith('_1'):
                slide_type_info[slide_name]['is_positive_slide'] = True

    available_slides = list(slide_to_positive_samples.keys())
    print(f"  发现包含阳性样本的slide: {len(available_slides)} 个")

    # 分类slide：明确的阳性slide（_1后缀）vs 其他slide
    explicit_positive_slides = [slide for slide in available_slides
                                if slide_type_info[slide]['is_positive_slide']]
    other_slides = [slide for slide in available_slides
                    if not slide_type_info[slide]['is_positive_slide']]

    print(f"  明确的阳性slide(_1后缀): {len(explicit_positive_slides)} 个")
    print(f"  其他包含阳性patch的slide: {len(other_slides)} 个")

    # 显示每个slide的阳性样本数量统计
    slide_counts = {slide: len(indices) for slide, indices in slide_to_positive_samples.items()}
    sorted_slides = sorted(slide_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"  阳性样本最多的5个slide:")
    for slide, count in sorted_slides[:5]:
        slide_type = "阳性slide" if slide.endswith('_1') else "其他"
        print(f"    {slide} ({slide_type}): {count} 个阳性patch")

    if len(available_slides) == 0:
        raise ValueError("没有找到包含阳性样本的slide！请检查数据和标签。")

    # 优先选择明确的阳性slide
    if len(explicit_positive_slides) >= num_slides:
        print(f"  ✅ 优先从 {len(explicit_positive_slides)} 个明确阳性slide中选择")
        selected_slides = random.sample(explicit_positive_slides, num_slides)
    elif len(explicit_positive_slides) > 0:
        print(f"  ⚠️ 明确阳性slide不足，将组合选择")
        remaining_needed = num_slides - len(explicit_positive_slides)
        if len(other_slides) >= remaining_needed:
            selected_from_others = random.sample(other_slides, remaining_needed)
            selected_slides = explicit_positive_slides + selected_from_others
        else:
            selected_slides = explicit_positive_slides + other_slides
        print(
            f"  选择: {len(explicit_positive_slides)}个明确阳性 + {len(selected_slides) - len(explicit_positive_slides)}个其他")
    else:
        print(f"  ⚠️ 没有明确的阳性slide(_1后缀)，从所有包含阳性patch的slide中选择")
        if len(available_slides) >= num_slides:
            selected_slides = random.sample(available_slides, num_slides)
        else:
            selected_slides = available_slides
            print(f"  将使用所有可用的 {len(available_slides)} 个slide")

    # 随机选择指定数量的slide
    random.seed(42)

    # 收集选中slide的所有阳性patch
    selected_positive_indices = []
    selected_slide_info = []

    for slide_name in selected_slides:
        slide_indices = slide_to_positive_samples[slide_name]
        selected_positive_indices.extend(slide_indices)
        slide_type = "_1" if slide_name.endswith('_1') else "其他"
        selected_slide_info.append(f"{slide_name}[{slide_type}]({len(slide_indices)}个)")

    print(f"  ✅ 选择结果:")
    print(f"    选中slide数: {len(selected_slides)}")
    print(f"    获得阳性patch总数: {len(selected_positive_indices)}")
    print(f"    选中的slide详情: {', '.join(selected_slide_info[:3])}...")  # 显示前3个
    print(f"    平均每slide阳性patch数: {len(selected_positive_indices) / len(selected_slides):.1f}")

    return selected_positive_indices


def get_camelyon16(args, alg='instant', dataset='camelyon16', num_labels=500, num_classes=2, data_dir='./data',
                   include_lb_to_ulb=True):
    """
    获取CAMELYON16数据集的SSL划分 - 官方划分版本
    """

    # 从args中获取参数
    if hasattr(args, 'data_dir'):
        data_dir = args.data_dir
    if hasattr(args, 'num_labels'):
        num_labels = args.num_labels
    if hasattr(args, 'include_lb_to_ulb'):
        include_lb_to_ulb = args.include_lb_to_ulb

    print(f"📦 初始化CAMELYON16数据集 (官方划分版本):")
    print(f"  数据目录: {data_dir}")
    print(f"  标记样本数: {num_labels}")
    print(f"  类别数: {num_classes}")

    # 检查数据目录
    if not os.path.exists(data_dir):
        raise ValueError(f"数据目录不存在: {data_dir}")

    # 参考CIFAR的数据增强策略
    crop_size = getattr(args, 'img_size', 224)
    crop_ratio = getattr(args, 'crop_ratio', 0.875)

    # 数据增强定义（参考cifar.py）
    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        # 按官方划分收集数据路径（自适应格式）
        max_samples = getattr(args, 'max_samples_per_slide', 200)

        (train_labeled_paths, train_labeled_targets, train_unlabeled_paths,
         test_paths, test_targets) = collect_camelyon16_paths_by_official_split(
            data_dir, max_samples_per_slide=max_samples
        )

        # 检查数据完整性
        if len(train_labeled_paths) == 0:
            raise ValueError(f"没有找到有标签的训练数据！请检查数据目录结构和文件命名规则。")

        class_1_count = train_labeled_targets.count(1)
        class_0_count = train_labeled_targets.count(0)

        if class_1_count == 0:
            raise ValueError(f"没有找到阳性样本！请检查是否有以'_1.jpg'或'_1.png'结尾的文件。\n" +
                             f"当前阴性样本数: {class_0_count}")

        if class_0_count == 0:
            raise ValueError(f"没有找到阴性样本！请检查是否有以'_0.jpg'或'_0.png'结尾的文件。\n" +
                             f"当前阳性样本数: {class_1_count}")

        # 新策略：固定选择slide数量，让数据量由实际情况决定
        print(f"\n🎯 实施更现实的标注策略")
        print("-" * 50)
        print(f"  原配置要求: {num_labels} 个标记样本")
        print(f"  新策略: 固定选择slide数量，适应实际数据分布")

        # 1. 固定选择10个阳性slide的所有阳性样本
        num_positive_slides = 10  # 固定选择10个slide
        selected_positive_indices = select_positive_samples_by_slide(
            train_labeled_paths, train_labeled_targets, num_positive_slides
        )

        actual_positive_count = len(selected_positive_indices)

        # 2. 阴性样本适配：选择与阳性样本相近的数量，或者固定一个合理数量
        # 策略A: 选择与阳性样本相同数量的阴性样本
        # 策略B: 选择阳性样本的2倍数量（模拟真实场景中阴性样本更多）
        # 策略C: 固定选择1000个阴性样本

        class_0_indices = [i for i, label in enumerate(train_labeled_targets) if label == 0]

        # 使用策略B：选择阳性样本2倍的阴性样本，但最多不超过2000个
        target_negative_count = min(actual_positive_count * 2, 2000, len(class_0_indices))

        if len(class_0_indices) < target_negative_count:
            print(f"  ⚠️ 阴性样本不足: 需要{target_negative_count}, 实际{len(class_0_indices)}")
            target_negative_count = len(class_0_indices)

        random.seed(42)
        selected_negative_indices = random.sample(class_0_indices, target_negative_count)

        actual_negative_count = len(selected_negative_indices)
        actual_total = actual_positive_count + actual_negative_count

        print(f"✅ 实际样本分布:")
        print(f"  选择的阳性slide数: {num_positive_slides}")
        print(f"  阳性样本数: {actual_positive_count}")
        print(f"  阴性样本数: {actual_negative_count}")
        print(f"  总标记样本数: {actual_total}")
        print(f"  阳性比例: {actual_positive_count / actual_total:.2%}")

        # 更新args中的num_labels为实际值
        args.num_labels = actual_total

        # 构建最终的标记集
        selected_labeled_indices = selected_negative_indices + selected_positive_indices
        remaining_labeled_indices = [i for i in range(len(train_labeled_paths))
                                     if i not in selected_labeled_indices]

        # 构建标记集（路径列表）
        lb_paths = [train_labeled_paths[i] for i in selected_labeled_indices]
        lb_targets = [train_labeled_targets[i] for i in selected_labeled_indices]

        # 构建无标记集（路径列表）
        ulb_paths = []
        ulb_targets = []

        # 添加剩余的有标签数据（隐藏标签）
        for i in remaining_labeled_indices:
            ulb_paths.append(train_labeled_paths[i])
            ulb_targets.append(train_labeled_targets[i])

        # 添加真正的无标签数据
        for path in train_unlabeled_paths:
            ulb_paths.append(path)
            ulb_targets.append(-1)  # -1表示真正的无标签

        # 如果include_lb_to_ulb=True，将标记数据也加入无标记集
        if include_lb_to_ulb:
            for i in selected_labeled_indices:
                ulb_paths.append(train_labeled_paths[i])
                ulb_targets.append(train_labeled_targets[i])

        print(f"📌 最终数据划分:")
        print(f"  标记集: {len(lb_paths)} 样本")
        print(f"    - 阴性: {lb_targets.count(0)}")
        print(f"    - 阳性: {lb_targets.count(1)}")
        print(f"  无标记集: {len(ulb_paths)} 样本")
        print(f"  测试集: {len(test_paths)} 样本")

        # 创建基于路径的数据集
        lb_dataset = PathBasedDataset(
            lb_paths, lb_targets, alg, num_classes, transform_weak,
            is_ulb=False, transform_strong=None
        )

        ulb_dataset = PathBasedDataset(
            ulb_paths, ulb_targets, alg, num_classes, transform_weak,
            is_ulb=True, transform_strong=transform_strong
        )

        # 使用官方测试集作为评估集
        eval_dataset = PathBasedDataset(
            test_paths, test_targets, alg, num_classes, transform_val,
            is_ulb=False, transform_strong=None
        )

        print("✅ CAMELYON16数据集创建成功 (官方划分版本)")

        return lb_dataset, ulb_dataset, eval_dataset

    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()

        # 提供详细的调试信息
        print(f"\n🔍 调试信息:")
        print(f"  数据目录: {data_dir}")
        print(f"  目录是否存在: {os.path.exists(data_dir)}")
        if os.path.exists(data_dir):
            subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            print(f"  子目录: {subdirs}")

        # 不要返回None，而是抛出异常
        raise e