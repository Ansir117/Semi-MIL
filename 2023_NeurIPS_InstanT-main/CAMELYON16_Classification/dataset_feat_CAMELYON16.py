import numpy as np
import torch
import torch.utils.data as data_utils
import os
import h5py
from tqdm import tqdm
import argparse

# 创建一个可导入的默认参数对象
args = argparse.Namespace()
args.seed = 500
args.max_patches = 1000
# 更新数据路径
args.train_data_path = '/home/xiaoyuan/Data3/CAMELYON16/training_feature'
args.test_data_path = '/home/xiaoyuan/Data3/CAMELYON16/testing_feature'
# 移除WSI限制，使用所有可用数据
args.train_wsi_limit = None


class PatchSamplingDataset(torch.utils.data.Dataset):
    """从每个幻灯片中采样固定数量的patches的包装数据集"""

    def __init__(self, original_dataset, max_patches=1000):
        self.dataset = original_dataset
        self.max_patches = max_patches

    def __getitem__(self, index):
        features, labels, idx = self.dataset[index]
        if isinstance(features, torch.Tensor) and features.shape[0] > self.max_patches:
            indices = torch.randperm(features.shape[0])[:self.max_patches]
            features = features[indices]
        elif isinstance(features, np.ndarray) and features.shape[0] > self.max_patches:
            indices = np.random.permutation(features.shape[0])[:self.max_patches]
            features = features[indices]
        return features, labels, idx

    def __len__(self):
        return len(self.dataset)


class Camelyon16_feat_dataset(torch.utils.data.Dataset):
    """Camelyon16数据集，使用预先划分的训练/测试集"""

    def __init__(self, train=True, return_bag=True, seed=42, max_patches=None,
                 train_data_path=None, test_data_path=None, train_wsi_limit=None):

        # 根据train参数选择数据路径
        if train:
            data_path = train_data_path if train_data_path else args.train_data_path
        else:
            data_path = test_data_path if test_data_path else args.test_data_path

        print(f"加载数据从: {data_path}")

        # 检查路径是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据路径不存在: {data_path}")

        # 加载所有特征和标签
        try:
            h5f = h5py.File(os.path.join(data_path, "patch_feat.h5"), 'r')
            self.all_patches = h5f['dataset_1'][:]
            h5f.close()
            print(f"成功加载特征文件，形状: {self.all_patches.shape}")
        except Exception as e:
            print(f"加载特征文件时出错: {e}")
            raise

        # 加载所有元数据
        try:
            self.patch_label = np.load(os.path.join(data_path, "patch_label.npy"))
            self.patch_corresponding_slide_label = np.load(
                os.path.join(data_path, "patch_corresponding_slide_label.npy"))
            self.patch_corresponding_slide_index = np.load(
                os.path.join(data_path, "patch_corresponding_slide_index.npy"))
            self.patch_corresponding_slide_name = np.load(
                os.path.join(data_path, "patch_corresponding_slide_name.npy"))
            self.patch_pos_x = np.load(os.path.join(data_path, "patch_pos_x.npy"))
            self.patch_pos_y = np.load(os.path.join(data_path, "patch_pos_y.npy"))
            print("成功加载所有元数据文件")
        except Exception as e:
            print(f"加载元数据文件时出错: {e}")
            raise

        # 获取所有唯一的幻灯片名称和索引
        all_slide_names = np.unique(self.patch_corresponding_slide_name)
        all_slide_indices = np.unique(self.patch_corresponding_slide_index)

        print(f"数据集中的总WSI数量: {len(all_slide_names)}")
        print(f"数据集中的总slide索引数量: {len(all_slide_indices)}")

        # 修改：移除WSI限制，使用所有可用数据
        # 如果是训练集且设置了WSI限制，则随机选择指定数量的WSI
        if train and train_wsi_limit and len(all_slide_indices) > train_wsi_limit:
            np.random.seed(seed)
            selected_slide_indices = np.random.choice(
                all_slide_indices, size=train_wsi_limit, replace=False
            )
            print(f"从训练集中随机选择 {train_wsi_limit} 个WSI进行训练")
        else:
            selected_slide_indices = all_slide_indices
            if train:
                print(f"使用所有 {len(selected_slide_indices)} 个训练WSI")
            else:
                print(f"使用所有 {len(selected_slide_indices)} 个测试WSI")

        # 根据选中的slide索引过滤patches
        selected_patches_mask = np.isin(self.patch_corresponding_slide_index, selected_slide_indices)

        # 更新所有数据
        self.all_patches = self.all_patches[selected_patches_mask]
        self.patch_label = self.patch_label[selected_patches_mask]
        self.patch_corresponding_slide_label = self.patch_corresponding_slide_label[selected_patches_mask]
        self.patch_corresponding_slide_index = self.patch_corresponding_slide_index[selected_patches_mask]
        self.patch_corresponding_slide_name = self.patch_corresponding_slide_name[selected_patches_mask]
        self.patch_pos_x = self.patch_pos_x[selected_patches_mask]
        self.patch_pos_y = self.patch_pos_y[selected_patches_mask]

        self.return_bag = return_bag
        self.num_patches = self.all_patches.shape[0]

        print(f"过滤后的patch数量: {self.num_patches}")

        # 处理位置信息
        self.patches_pos = np.stack((self.patch_pos_x, self.patch_pos_y), axis=1)

        # 按幻灯片组织特征
        self.slide_feat_all = []
        self.slide_label_all = []
        self.slide_patch_label_all = []
        self.slide_name_all = []
        self.slide_pos_all = []

        # 获取过滤后数据中的所有唯一幻灯片索引
        unique_slide_indices = np.unique(self.patch_corresponding_slide_index)

        for slide_idx in tqdm(unique_slide_indices, desc='组织幻灯片数据'):
            idx_from_same_slide = np.where(self.patch_corresponding_slide_index == slide_idx)[0]

            if len(idx_from_same_slide) == 0:
                continue

            # 处理位置信息
            bag_patch_pos_x = self.patch_pos_x[idx_from_same_slide]
            bag_patch_pos_y = self.patch_pos_y[idx_from_same_slide]
            bag_patch_pos = np.concatenate((bag_patch_pos_y.reshape(1, -1),
                                            bag_patch_pos_x.reshape(1, -1)))

            # 获取特征和标签
            bag = self.all_patches[idx_from_same_slide]
            slide_labels = self.patch_corresponding_slide_label[idx_from_same_slide]
            slide_names = self.patch_corresponding_slide_name[idx_from_same_slide]

            # 检查标签一致性
            unique_labels = np.unique(slide_labels)
            if len(unique_labels) != 1:
                print(f"警告: 在幻灯片 {slide_idx} 中发现不一致的标签: {unique_labels}")
                continue

            # 检查slide名称一致性
            unique_names = np.unique(slide_names)
            if len(unique_names) != 1:
                print(f"警告: 在幻灯片 {slide_idx} 中发现不一致的名称: {unique_names}")
                continue

            # 如果指定了max_patches，应用限制
            if max_patches is not None and bag.shape[0] > max_patches:
                np.random.seed(seed + slide_idx)  # 确保可重现性
                indices = np.random.choice(bag.shape[0], max_patches, replace=False)
                bag = bag[indices]
                bag_patch_pos = bag_patch_pos[:, indices]

            self.slide_feat_all.append(bag)
            self.slide_label_all.append(slide_labels[0])
            self.slide_patch_label_all.append(self.patch_label[idx_from_same_slide].astype(np.int64))
            self.slide_name_all.append(slide_names[0])
            self.slide_pos_all.append(bag_patch_pos)

        # 打印数据集统计信息
        self._print_dataset_stats(train)

    def _print_dataset_stats(self, is_train=True):
        """打印数据集统计信息"""
        dataset_type = "训练" if is_train else "测试"
        print(f"\n[{dataset_type}数据信息] 成功加载 {len(self.slide_feat_all)} 张幻灯片")

        if len(self.slide_label_all) > 0:
            labels = np.array(self.slide_label_all)
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"标签分布: {dict(zip(unique_labels, counts))}")
            for label, count in zip(unique_labels, counts):
                label_name = "negative" if label == 0 else "positive"
                print(f"类别 {label} ({label_name}): {count} 张幻灯片 ({count / len(labels) * 100:.2f}%)")

            # 基于频率计算类别权重
            total_slides = len(labels)
            class_weights = [total_slides / (len(unique_labels) * count) for count in counts]
            self.class_weights = torch.FloatTensor(class_weights)
            print(f"类别权重: {self.class_weights}")
        else:
            print("警告: 未加载任何幻灯片")
            self.class_weights = torch.FloatTensor([1.0, 1.0])  # 默认相等权重

        # 打印特征统计信息
        if len(self.slide_feat_all) > 0:
            sample_feat = self.slide_feat_all[0]
            print(f"特征形状: {sample_feat.shape}")
            print(f"特征维度: {sample_feat.shape[1]}")

            # 打印一些样本的patch数量
            patch_counts = [feat.shape[0] for feat in self.slide_feat_all]
            print(f"每个WSI的patch数量统计:")
            print(f"  最小: {min(patch_counts)}")
            print(f"  最大: {max(patch_counts)}")
            print(f"  平均: {np.mean(patch_counts):.1f}")
            print(f"  中位数: {np.median(patch_counts):.1f}")

    def __getitem__(self, index):
        if self.return_bag:
            return (self.slide_feat_all[index],
                    [self.slide_patch_label_all[index],
                     self.slide_label_all[index],
                     self.slide_name_all[index],
                     self.slide_pos_all[index]],
                    index)
        else:
            return (self.all_patches[index],
                    [self.patch_label[index],
                     self.patch_corresponding_slide_label[index],
                     self.patch_corresponding_slide_name[index],
                     [self.patch_pos_y[index], self.patch_pos_x[index]]],
                    index)

    def __len__(self):
        if self.return_bag:
            return len(self.slide_feat_all)
        else:
            return self.num_patches


def prepare_data(args):
    """准备训练和测试数据集"""
    print("=" * 60)
    print("准备Camelyon16数据集...")
    print("=" * 60)

    train_dataset = Camelyon16_feat_dataset(
        train=True,
        return_bag=True,
        seed=args.seed,
        max_patches=args.max_patches,
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        train_wsi_limit=args.train_wsi_limit
    )

    test_dataset = Camelyon16_feat_dataset(
        train=False,
        return_bag=True,
        seed=args.seed,
        max_patches=args.max_patches,
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        train_wsi_limit=None  # 测试集不限制数量
    )

    # 如果设置了max_patches，则使用PatchSamplingDataset包装数据集
    if args.max_patches > 0:
        train_dataset = PatchSamplingDataset(train_dataset, max_patches=args.max_patches)
        test_dataset = PatchSamplingDataset(test_dataset, max_patches=args.max_patches)

    validate_dataset(train_dataset, "训练")
    validate_dataset(test_dataset, "测试")

    return train_dataset, test_dataset


def validate_dataset(dataset, name=""):
    """验证数据集完整性和分布"""
    if hasattr(dataset, 'dataset'):
        slide_labels = np.array(dataset.dataset.slide_label_all)
    else:
        slide_labels = np.array(dataset.slide_label_all)

    unique_labels, counts = np.unique(slide_labels, return_counts=True)

    print(f"\n{name}数据集统计信息:")
    print(f"总幻灯片数: {len(dataset)}")
    print(f"类别分布: {dict(zip(unique_labels, counts))}")

    # 打印更详细的类别信息
    for label, count in zip(unique_labels, counts):
        label_name = "negative" if label == 0 else "positive"
        print(f"  类别 {label} ({label_name}): {count} 张幻灯片 ({count / len(slide_labels) * 100:.2f}%)")

    # 计算特征统计信息
    if hasattr(dataset, 'dataset'):
        all_features = np.vstack([feat for feat in dataset.dataset.slide_feat_all])
    else:
        all_features = np.vstack([feat for feat in dataset.slide_feat_all])

    print(f"特征统计信息:")
    print(f"特征维度: {all_features.shape}")
    print(f"特征平均值: {np.mean(all_features):.3f}")
    print(f"特征标准差: {np.std(all_features):.3f}")
    print(f"特征范围: [{np.min(all_features):.3f}, {np.max(all_features):.3f}]")


# 用于测试
if __name__ == '__main__':
    # 只在直接运行该脚本时解析命令行参数
    parser = argparse.ArgumentParser(description='测试Camelyon16数据集加载')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--max_patches', default=1000, type=int, help='每张幻灯片的最大patches数')
    parser.add_argument('--train_data_path',
                        default='/home/xiaoyuan/Data3/CAMELYON16/training_feature',
                        type=str, help='训练集特征数据路径')
    parser.add_argument('--test_data_path',
                        default='/home/xiaoyuan/Data3/CAMELYON16/testing_feature',
                        type=str, help='测试集特征数据路径')
    parser.add_argument('--train_wsi_limit', default=None, type=int, help='训练集WSI数量限制（None表示不限制）')

    cmd_args = parser.parse_args()

    # 更新全局默认参数
    args.seed = cmd_args.seed
    args.max_patches = cmd_args.max_patches
    args.train_data_path = cmd_args.train_data_path
    args.test_data_path = cmd_args.test_data_path
    args.train_wsi_limit = cmd_args.train_wsi_limit

    # 测试数据集加载
    print("测试训练数据集加载...")
    train_dataset = Camelyon16_feat_dataset(
        train=True,
        return_bag=True,
        seed=args.seed,
        max_patches=args.max_patches,
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        train_wsi_limit=args.train_wsi_limit
    )

    print("\n测试测试数据集加载...")
    test_dataset = Camelyon16_feat_dataset(
        train=False,
        return_bag=True,
        seed=args.seed,
        max_patches=args.max_patches,
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        train_wsi_limit=None
    )

    print(f"\n最终结果:")
    print(f"训练数据集大小: {len(train_dataset)}")
    print(f"测试数据集大小: {len(test_dataset)}")

    # 测试加载一个样本
    print("\n测试样本加载...")
    try:
        sample_data, sample_label, sample_idx = train_dataset[0]
        print(f"样本特征形状: {sample_data.shape}")
        print(f"样本标签: {sample_label[1]}")
        print(f"样本名称: {sample_label[2]}")
        print("样本加载成功!")
    except Exception as e:
        print(f"样本加载失败: {e}")