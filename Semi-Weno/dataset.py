"""
CAMELYON16 端到端数据集 - 支持WENO+SSL混合训练
==============================================
整合图像加载、特征提取、SSL标签划分于一体

数据划分逻辑：
1. Teacher训练：使用所有slides（bag级别有标签）
2. SSL分类器：
   - 监督部分：从选定slides中采样的有patch标签的数据
   - 无监督部分：其余所有patches
3. Student训练：使用所有patches + 混合伪标签

数据目录结构：
    data_dir/
    ├── training/
    │   ├── tumor_001_1/  或 slide_001_pos/  (阳性slide)
    │   │   ├── patch_0_1.jpg  或 xxx_pos.jpg (阳性patch)
    │   │   ├── patch_1_0.jpg  或 xxx_neg.jpg (阴性patch)
    │   │   └── patch_2.jpg    (无标注patch)
    │   └── normal_001_0/ 或 slide_002_neg/  (阴性slide)
    └── testing/
        └── (同上)
"""

import os
import glob
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import random


class CAMELYON16_E2E(Dataset):
    """
    CAMELYON16端到端数据集

    Args:
        root_dir: 数据根目录
        train: True=training, False=testing
        transform: 图像变换
        return_bag: True返回整个slide，False返回单个patch
        max_bag_size: bag模式下最大patch数量
        preload: 是否预加载图像到内存
    """

    def __init__(self, root_dir, train=True, transform=None, return_bag=False,
                 max_bag_size=100, preload=False, patch_label_file=""):

        self.root_dir = root_dir
        self.train = train
        self.return_bag = return_bag
        self.max_bag_size = max_bag_size
        self.preload = preload
        self.patch_label_file = patch_label_file
        self.patch_label_map = self._load_patch_label_file(patch_label_file)

        # 默认transform
        if transform is None:
            if train:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

        # 加载数据
        self._load_data()

    def _get_slide_label(self, slide_name):
        """判断slide标签"""
        name_lower = slide_name.lower()
        if 'pos' in name_lower or 'tumor' in name_lower or slide_name.endswith('_1'):
            return 1
        return 0

    def _get_patch_label(self, patch_name, slide_label):
        """判断patch标签"""
        base = os.path.splitext(patch_name)[0].lower()

        if '_pos' in base or base.endswith('_1'):
            return 1
        elif '_neg' in base or base.endswith('_0'):
            return 0
        else:
            # 无明确标注
            return 0 if slide_label == 0 else -1

    @staticmethod
    def _load_patch_label_file(patch_label_file):
        if patch_label_file is None or patch_label_file == "":
            return {}
        if not os.path.exists(patch_label_file):
            raise FileNotFoundError(f"patch_label_file not found: {patch_label_file}")

        patch_label_map = {}
        with open(patch_label_file, "r", newline="") as f:
            sample = f.read(2048)
            f.seek(0)
            delimiter = "," if sample.count(",") >= sample.count("\t") else "\t"
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                if len(row) < 2:
                    continue
                key = row[0].strip()
                value = row[1].strip()
                if key == "" or value == "":
                    continue
                if value.lower() in ["label", "patch_label"]:
                    continue
                patch_label_map[os.path.normpath(key)] = int(float(value))
        return patch_label_map

    def _resolve_patch_label_from_file(self, img_path, slide_name, patch_name):
        if len(self.patch_label_map) == 0:
            return None
        rel_key = os.path.join(slide_name, patch_name)
        candidates = [
            os.path.normpath(img_path),
            os.path.normpath(rel_key),
            os.path.normpath(patch_name),
        ]
        for key in candidates:
            if key in self.patch_label_map:
                return int(self.patch_label_map[key])
        return None

    def _load_data(self):
        """加载数据路径和标签"""
        data_dir = os.path.join(self.root_dir, 'training' if self.train else 'testing')

        if not os.path.exists(data_dir):
            raise ValueError(f"数据目录不存在: {data_dir}")

        # 获取所有slide目录
        all_slides = sorted([d for d in os.listdir(data_dir)
                             if os.path.isdir(os.path.join(data_dir, d))])

        self.all_paths = []
        self.patch_labels = []
        self.patch_label_has_gt = []
        self.slide_labels = []
        self.slide_indices = []
        self.slide_names = []

        print(f"[DATA] 加载{'训练' if self.train else '测试'}数据...")

        for slide_idx, slide_name in enumerate(tqdm(all_slides, desc='扫描slides')):
            slide_dir = os.path.join(data_dir, slide_name)
            slide_label = self._get_slide_label(slide_name)

            # 获取所有图像文件
            img_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif']:
                img_files.extend(glob.glob(os.path.join(slide_dir, ext)))
                img_files.extend(glob.glob(os.path.join(slide_dir, ext.upper())))

            for img_path in img_files:
                patch_name = os.path.basename(img_path)
                patch_label = self._get_patch_label(patch_name, slide_label)
                patch_label_from_file = self._resolve_patch_label_from_file(img_path, slide_name, patch_name)
                if patch_label_from_file is not None:
                    patch_label = patch_label_from_file
                    patch_has_gt = 1
                else:
                    patch_has_gt = 1 if patch_label >= 0 else 0

                self.all_paths.append(img_path)
                self.patch_labels.append(patch_label)
                self.patch_label_has_gt.append(patch_has_gt)
                self.slide_labels.append(slide_label)
                self.slide_indices.append(slide_idx)
                self.slide_names.append(slide_name)

        # 转换为numpy数组
        self.all_paths = np.array(self.all_paths)
        self.patch_labels = np.array(self.patch_labels)
        self.patch_label_has_gt = np.array(self.patch_label_has_gt)
        self.slide_labels = np.array(self.slide_labels)
        self.slide_indices = np.array(self.slide_indices)
        self.slide_names = np.array(self.slide_names)

        self.num_slides = len(all_slides)
        self.num_patches = len(self.all_paths)

        # 预加载（可选）
        if self.preload:
            print("[DATA] 预加载图像到内存...")
            self.images = []
            for path in tqdm(self.all_paths, desc='预加载'):
                img = Image.open(path).convert('RGB')
                self.images.append(img)

        # 打印统计
        print(f"[DATA] {self.num_slides} slides, {self.num_patches} patches")
        print(f"[DATA] 阳性patch: {(self.patch_labels == 1).sum()}, "
              f"阴性patch: {(self.patch_labels == 0).sum()}, "
              f"无标注: {(self.patch_labels == -1).sum()}")
        print(f"[DATA] 真实patch标注数量: {int(self.patch_label_has_gt.sum())}")

    def _load_image(self, index):
        """加载单张图像"""
        if self.preload:
            img = self.images[index].copy()
        else:
            img = Image.open(self.all_paths[index]).convert('RGB')
        return img

    def __len__(self):
        return self.num_slides if self.return_bag else self.num_patches

    def __getitem__(self, index):
        if self.return_bag:
            return self._get_bag(index)
        else:
            return self._get_patch(index)

    def _get_patch(self, index):
        """返回单个patch"""
        img = self._load_image(index)
        if self.transform:
            img = self.transform(img)

        patch_label = self.patch_labels[index]
        slide_label = self.slide_labels[index]
        slide_idx = self.slide_indices[index]
        slide_name = self.slide_names[index]

        return img, [torch.tensor(patch_label), torch.tensor(slide_label),
                     torch.tensor(slide_idx), slide_name,
                     torch.tensor(self.patch_label_has_gt[index])], index

    def _get_bag(self, slide_index):
        """返回整个slide（bag）"""
        patch_indices = np.where(self.slide_indices == slide_index)[0]

        # 限制bag大小
        if len(patch_indices) > self.max_bag_size:
            patch_indices = np.random.choice(patch_indices, self.max_bag_size, replace=False)

        # 加载所有patch
        images = []
        for idx in patch_indices:
            img = self._load_image(idx)
            if self.transform:
                img = self.transform(img)
            images.append(img)

        images = torch.stack(images)  # [N, C, H, W]
        patch_labels = torch.tensor(self.patch_labels[patch_indices])
        patch_label_has_gt = torch.tensor(self.patch_label_has_gt[patch_indices])
        slide_label = torch.tensor(self.slide_labels[patch_indices[0]])
        slide_name = self.slide_names[patch_indices[0]]

        return images, [patch_labels, slide_label, slide_index, slide_name, patch_label_has_gt], slide_index


def create_ssl_split(dataset, labeled_slide_ratio=0.2, labeled_patch_ratio=0.5, seed=42):
    """
    创建半监督学习的数据划分

    Args:
        dataset: CAMELYON16_E2E数据集实例
        labeled_slide_ratio: 有标签slide的比例
        labeled_patch_ratio: 每个有标签slide中采样的patch比例
        seed: 随机种子

    Returns:
        dict: {
            'labeled_indices': 有标签patch的索引,
            'unlabeled_indices': 无标签patch的索引,
            'labeled_slides': 被选为有标签的slide名称
        }
    """
    np.random.seed(seed)

    # 获取所有唯一的slide
    unique_slides = np.unique(dataset.slide_indices)

    # 分开阳性和阴性slides
    pos_slides = []
    neg_slides = []
    for s in unique_slides:
        mask = dataset.slide_indices == s
        if dataset.slide_labels[mask][0] == 1:
            pos_slides.append(s)
        else:
            neg_slides.append(s)

    # 按比例选择有标签的slides
    num_labeled_pos = max(1, int(len(pos_slides) * labeled_slide_ratio))
    num_labeled_neg = max(1, int(len(neg_slides) * labeled_slide_ratio))

    labeled_pos_slides = np.random.choice(pos_slides, num_labeled_pos, replace=False)
    labeled_neg_slides = np.random.choice(neg_slides, num_labeled_neg, replace=False)
    labeled_slides = np.concatenate([labeled_pos_slides, labeled_neg_slides])

    print(f"[SSL划分] 选择 {len(labeled_slides)} 个有标签slides "
          f"({len(labeled_pos_slides)} 阳性, {len(labeled_neg_slides)} 阴性)")

    # 从有标签slides中采样patches
    labeled_indices = []
    # 优先纳入真实标注patch（你提供的patch标签）
    if hasattr(dataset, 'patch_label_has_gt'):
        gt_indices = np.where((dataset.patch_label_has_gt == 1) & (dataset.patch_labels >= 0))[0]
        if len(gt_indices) > 0:
            labeled_indices.extend(gt_indices.tolist())
    for slide_idx in labeled_slides:
        slide_mask = dataset.slide_indices == slide_idx
        slide_patch_indices = np.where(slide_mask)[0]

        # 只选择有明确标签的patches (不是-1)
        valid_mask = dataset.patch_labels[slide_patch_indices] >= 0
        valid_indices = slide_patch_indices[valid_mask]

        if len(valid_indices) > 0:
            num_to_sample = max(1, int(len(valid_indices) * labeled_patch_ratio))
            sampled = np.random.choice(valid_indices, min(num_to_sample, len(valid_indices)), replace=False)
            labeled_indices.extend(sampled)

    labeled_indices = np.array(sorted(list(set(labeled_indices))))

    # 其余所有patch作为无标签数据
    all_indices = np.arange(dataset.num_patches)
    unlabeled_indices = np.setdiff1d(all_indices, labeled_indices)

    # 统计
    labeled_pos = (dataset.patch_labels[labeled_indices] == 1).sum()
    labeled_neg = (dataset.patch_labels[labeled_indices] == 0).sum()

    print(f"[SSL划分] 有标签patches: {len(labeled_indices)} "
          f"(阳性:{labeled_pos}, 阴性:{labeled_neg})")
    print(f"[SSL划分] 无标签patches: {len(unlabeled_indices)}")

    # 获取有标签slides的名称
    labeled_slide_names = [dataset.slide_names[dataset.slide_indices == s][0]
                           for s in labeled_slides]

    return {
        'labeled_indices': labeled_indices,
        'unlabeled_indices': unlabeled_indices,
        'labeled_slides': labeled_slide_names,
        'num_labeled': len(labeled_indices),
        'num_unlabeled': len(unlabeled_indices)
    }


class SSLPatchDataset(Dataset):
    """
    用于SSL分类器训练的数据集包装器
    区分有标签和无标签数据
    """

    def __init__(self, base_dataset, labeled_indices, unlabeled_indices,
                 transform_weak=None, transform_strong=None):
        """
        Args:
            base_dataset: 基础数据集
            labeled_indices: 有标签数据的索引
            unlabeled_indices: 无标签数据的索引
            transform_weak: 弱增强
            transform_strong: 强增强（用于无标签数据的一致性正则化）
        """
        self.base_dataset = base_dataset
        self.labeled_indices = np.array(labeled_indices)
        self.unlabeled_indices = np.array(unlabeled_indices)
        self.transform_weak = transform_weak or base_dataset.transform
        self.transform_strong = transform_strong or self.transform_weak

        # 合并所有索引，但记录哪些是有标签的
        self.all_indices = np.concatenate([self.labeled_indices, self.unlabeled_indices])
        self.is_labeled = np.concatenate([
            np.ones(len(self.labeled_indices), dtype=bool),
            np.zeros(len(self.unlabeled_indices), dtype=bool)
        ])

    def __len__(self):
        return len(self.all_indices)

    def __getitem__(self, idx):
        real_idx = self.all_indices[idx]
        is_labeled = self.is_labeled[idx]

        # 加载图像
        img = self.base_dataset._load_image(real_idx)

        patch_label = self.base_dataset.patch_labels[real_idx]
        slide_label = self.base_dataset.slide_labels[real_idx]
        slide_idx = self.base_dataset.slide_indices[real_idx]

        if is_labeled:
            # 有标签数据：只返回弱增强
            img_w = self.transform_weak(img)
            has_gt = self.base_dataset.patch_label_has_gt[real_idx] if hasattr(self.base_dataset, 'patch_label_has_gt') else 0
            return {
                'image': img_w,
                'patch_label': torch.tensor(patch_label),
                'slide_label': torch.tensor(slide_label),
                'slide_idx': torch.tensor(slide_idx),
                'is_labeled': torch.tensor(True),
                'has_gt': torch.tensor(has_gt),
                'index': real_idx
            }
        else:
            # 无标签数据：返回弱增强和强增强
            img_w = self.transform_weak(img)
            img_s = self.transform_strong(img)
            has_gt = self.base_dataset.patch_label_has_gt[real_idx] if hasattr(self.base_dataset, 'patch_label_has_gt') else 0
            return {
                'image_w': img_w,
                'image_s': img_s,
                'patch_label': torch.tensor(patch_label),  # 用于评估，训练时不用
                'slide_label': torch.tensor(slide_label),
                'slide_idx': torch.tensor(slide_idx),
                'is_labeled': torch.tensor(False),
                'has_gt': torch.tensor(has_gt),
                'index': real_idx
            }


# ==================== 兼容原WENO的类名 ====================
class CAMELYON_16(CAMELYON16_E2E):
    """兼容原WENO代码的类名"""

    def __init__(self, root_dir='', train=True, transform=None, downsample=1.0,
                 drop_threshold=0.0, preload=False, return_bag=False, patch_label_file=""):
        super().__init__(
            root_dir=root_dir, train=train, transform=transform,
            return_bag=return_bag, preload=preload, patch_label_file=patch_label_file
        )


# ==================== 测试代码 ====================
if __name__ == '__main__':
    print("=" * 50)
    print("测试 CAMELYON16_E2E 数据集")
    print("=" * 50)

    # 修改为您的数据路径
    data_dir = '/path/to/CAMELYON16'

    if os.path.exists(data_dir):
        # 创建基础数据集
        ds = CAMELYON16_E2E(data_dir, train=True, return_bag=False)

        # 创建SSL划分
        ssl_split = create_ssl_split(ds, labeled_slide_ratio=0.2, labeled_patch_ratio=0.5)

        # 创建SSL数据集
        ssl_ds = SSLPatchDataset(ds, ssl_split['labeled_indices'], ssl_split['unlabeled_indices'])
        print(f"\nSSL数据集大小: {len(ssl_ds)}")

        # 测试取数据
        sample = ssl_ds[0]  # 有标签
        print(f"有标签样本: is_labeled={sample['is_labeled']}, label={sample['patch_label']}")

        sample = ssl_ds[len(ssl_split['labeled_indices']) + 1]  # 无标签
        print(f"无标签样本: is_labeled={sample['is_labeled']}")
    else:
        print(f"数据目录不存在: {data_dir}")