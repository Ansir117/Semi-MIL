import os
import glob
import numpy as np
import random
from skimage import io
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class Camelyon16_Dataset(Dataset):
    def __init__(self, slides, transform=None, return_bag=False):
        self.slides = slides
        self.transform = transform
        self.return_bag = return_bag

        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

        self.all_patches = []  # 存储所有patch的路径
        self.patch_label = []  # patch级别的标签（全部设为0）
        self.patch_corresponding_slide_label = []  # 对应slide的标签(0或1)
        self.patch_corresponding_slide_index = []  # patch对应的slide索引
        self.patch_corresponding_slide_name = []  # patch对应的slide名称
        self.patch_pos_x = []  # patch的x坐标
        self.patch_pos_y = []  # patch的y坐标

        cnt_slide = 0
        cnt_patch = 0

        for slide in tqdm(self.slides, ascii=True, desc='preload data'):
            try:
                slide_name = os.path.basename(slide)  # 获取文件夹名称
                # 新的命名格式: test_001_1, tumor_001_0 等，最后一位是标签
                parts = slide_name.split('_')
                if len(parts) >= 3:
                    label_str = parts[-1]  # 获取最后一个部分作为标签
                    try:
                        label = int(label_str)  # 直接转换为整数
                        if label not in [0, 1]:
                            print(
                                f"Warning: Invalid label '{label}' found in slide {slide_name} (must be 0 or 1), skipping...")
                            continue
                    except ValueError:
                        print(
                            f"Warning: Cannot convert label '{label_str}' to integer in slide {slide_name}, skipping...")
                        continue
                else:
                    print(
                        f"Warning: Invalid naming format in slide {slide_name}, expected format: prefix_number_label, skipping...")
                    continue

                # 打印前几个slide的信息，用于调试
                if cnt_slide < 3:
                    print(f"Processing slide: {slide_name} with label: {label}")

            except Exception as e:
                print(f"Error parsing label from slide {slide_name}: {e}")
                continue

            # 获取当前slide文件夹中的所有文件
            patch_files = []
            try:
                for j in os.listdir(slide):
                    file_path = os.path.join(slide, j)
                    # 确保只添加图像文件，不添加子文件夹
                    if os.path.isfile(file_path) and (
                            j.lower().endswith('.jpg') or j.lower().endswith('.png') or j.lower().endswith('.jpeg')):
                        patch_files.append(j)
                    elif os.path.isdir(file_path):
                        print(f"Warning: Skipping subfolder {j} in slide {slide_name}")
            except Exception as e:
                print(f"Error listing files in slide {slide_name}: {e}")
                continue

            # 如果没有找到有效图像文件，跳过此slide
            if not patch_files:
                print(f"Warning: No valid image files found in slide {slide_name}, skipping...")
                continue

            # 处理每个patch文件
            for j in patch_files:
                file_path = os.path.join(slide, j)
                self.all_patches.append(file_path)
                self.patch_label.append(0)  # patch级别标签统一设为0
                self.patch_corresponding_slide_label.append(label)
                self.patch_corresponding_slide_index.append(cnt_slide)
                self.patch_corresponding_slide_name.append(slide_name)

                # 从文件名解析坐标
                try:
                    # 如果是第一个文件，打印一个示例以供调试
                    if cnt_patch == 0:
                        print(f"Debug - Patch filename example: {j}")

                    # 根据patch文件名格式，通常是 "x_y.jpg" 或类似格式
                    filename_parts = j.split('.')
                    name_without_ext = filename_parts[0]  # 去掉扩展名

                    # 尝试解析坐标，可能的格式包括：
                    # 1. "x_y.jpg"
                    # 2. "patch_x_y.jpg"
                    # 3. 其他格式
                    if '_' in name_without_ext:
                        coord_parts = name_without_ext.split('_')
                        if len(coord_parts) >= 2:
                            # 取最后两个数字作为坐标
                            try:
                                patch_pos_x = int(coord_parts[-2])  # 倒数第二个
                                patch_pos_y = int(coord_parts[-1])  # 最后一个
                            except ValueError:
                                # 如果不能转换为数字，使用前两个
                                patch_pos_x = int(coord_parts[0]) if coord_parts[0].isdigit() else cnt_patch % 1000
                                patch_pos_y = int(coord_parts[1]) if coord_parts[1].isdigit() else cnt_patch // 1000
                        else:
                            # 如果格式不符合预期，使用默认值
                            patch_pos_x = cnt_patch % 1000
                            patch_pos_y = cnt_patch // 1000
                            if cnt_patch == 0:
                                print(f"Warning: Unexpected filename format: {j}, using default coordinates")
                    else:
                        # 没有下划线，使用默认坐标
                        patch_pos_x = cnt_patch % 1000
                        patch_pos_y = cnt_patch // 1000
                        if cnt_patch == 0:
                            print(f"Warning: No underscore in filename {j}, using default coordinates")

                except Exception as e:
                    # 捕获任何异常，使用安全的默认值
                    if cnt_patch == 0:
                        print(f"Warning: Could not parse coordinates from filename {j}, using default. Error: {e}")
                    patch_pos_x = cnt_patch % 1000
                    patch_pos_y = cnt_patch // 1000

                self.patch_pos_x.append(patch_pos_x)
                self.patch_pos_y.append(patch_pos_y)
                cnt_patch += 1

            cnt_slide += 1

        self.num_patches = cnt_patch
        self.all_patches = np.array(self.all_patches)
        self.patch_label = np.array(self.patch_label, dtype=np.int64)  # 确保整数类型
        self.patch_corresponding_slide_label = np.array(self.patch_corresponding_slide_label, dtype=np.int64)  # 确保整数类型
        self.patch_corresponding_slide_index = np.array(self.patch_corresponding_slide_index, dtype=np.int64)  # 确保整数类型
        self.patch_corresponding_slide_name = np.array(self.patch_corresponding_slide_name)
        self.patch_pos_x = np.array(self.patch_pos_x, dtype=np.int64)  # 确保整数类型
        self.patch_pos_y = np.array(self.patch_pos_y, dtype=np.int64)  # 确保整数类型

        # 添加标签验证
        unique_labels = np.unique(self.patch_corresponding_slide_label)
        print(f"\nUnique labels in dataset: {unique_labels}")
        if not all(label in range(2) for label in unique_labels):  # 2分类，所以标签应该是0或1
            raise ValueError("Dataset contains invalid labels. All labels should be in range [0, 1]")
        print(f"Total patches loaded: {self.num_patches}")

        # 确保使用整数数组进行bincount
        try:
            dist = np.bincount(self.patch_corresponding_slide_label)
            print(f"Class distribution: {dist} (0:negative, 1:positive)")
        except TypeError:
            print(
                f"Warning: Cannot compute class distribution, label type: {self.patch_corresponding_slide_label.dtype}")
            print(f"Converting labels to integers...")
            int_labels = self.patch_corresponding_slide_label.astype(np.int64)
            print(f"Class distribution: {np.bincount(int_labels)} (0:negative, 1:positive)")

        print("")

    def __getitem__(self, index):
        if self.return_bag:
            idx_patch_from_slide_i = np.where(self.patch_corresponding_slide_index == index)[0]
            bag = self.all_patches[idx_patch_from_slide_i]
            bag_normed = np.zeros([bag.shape[0], 3, 224, 224], dtype=np.float32)
            for i in range(bag.shape[0]):
                try:
                    # 确保路径是图像文件而不是文件夹
                    if not os.path.isfile(bag[i]):
                        print(f"Warning: {bag[i]} is not a file, skipping...")
                        # 使用零矩阵替代
                        continue

                    instance_img = io.imread(bag[i])
                    bag_normed[i, :, :, :] = self.transform(Image.fromarray(np.uint8(instance_img), 'RGB'))
                except Exception as e:
                    print(f"Error reading image {bag[i]}: {e}")
                    # 使用零矩阵替代
                    continue
            patch_labels = self.patch_label[idx_patch_from_slide_i]
            slide_label = self.patch_corresponding_slide_label[idx_patch_from_slide_i][0]
            slide_index = self.patch_corresponding_slide_index[idx_patch_from_slide_i][0]
            slide_name = self.patch_corresponding_slide_name[idx_patch_from_slide_i][0]
            return bag_normed, [patch_labels, slide_label, slide_index, slide_name], index
        else:
            try:
                # 确保路径是图像文件而不是文件夹
                patch_path = self.all_patches[index]
                if not os.path.isfile(patch_path):
                    print(f"Warning: {patch_path} is not a file, using empty image instead")
                    # 返回空图像
                    empty_image = np.zeros((224, 224, 3), dtype=np.uint8)
                    patch_image = self.transform(Image.fromarray(empty_image, 'RGB'))
                else:
                    patch_image = io.imread(patch_path)
                    patch_image = self.transform(Image.fromarray(np.uint8(patch_image), 'RGB'))

                return patch_image, [self.patch_label[index], self.patch_corresponding_slide_label[index],
                                     self.patch_corresponding_slide_index[index],
                                     self.patch_corresponding_slide_name[index]], index
            except Exception as e:
                print(f"Error in __getitem__ for index {index}, path {self.all_patches[index]}: {e}")
                # 返回空图像
                empty_image = np.zeros((224, 224, 3), dtype=np.uint8)
                patch_image = self.transform(Image.fromarray(empty_image, 'RGB'))
                return patch_image, [self.patch_label[index], self.patch_corresponding_slide_label[index],
                                     self.patch_corresponding_slide_index[index],
                                     self.patch_corresponding_slide_name[index]], index

    def __len__(self):
        return self.num_patches if not self.return_bag else self.patch_corresponding_slide_index.max() + 1


def statistics_slides(directory_path):
    """统计数据集信息"""
    neg_count = 0  # 标签为0
    pos_count = 0  # 标签为1

    for slide_path in glob.glob(os.path.join(directory_path, "*")):
        if not os.path.isdir(slide_path):
            continue

        slide_name = os.path.basename(slide_path)
        try:
            parts = slide_name.split('_')
            if len(parts) >= 3:
                label_str = parts[-1]
                label = int(label_str)
                if label == 0:
                    neg_count += 1
                elif label == 1:
                    pos_count += 1
                else:
                    print(f"Warning: Unexpected label '{label}' in slide {slide_name}")
            else:
                print(f"Warning: Invalid naming format in slide {slide_name}")
        except Exception as e:
            print(f"Warning: Cannot parse label from slide {slide_name}: {e}")

    total_slides = neg_count + pos_count
    if total_slides > 0:
        print(f"[DATA INFO] Total number of slides: {total_slides}")
        print(f"[DATA INFO] Negative slides (label 0): {neg_count} ({neg_count / total_slides * 100:.2f}%)")
        print(f"[DATA INFO] Positive slides (label 1): {pos_count} ({pos_count / total_slides * 100:.2f}%)")
    else:
        print("[DATA INFO] No valid slides found!")


def get_dataloader(root_dir='/home/xiaoyuan/Data3/CAMELYON16/testing', batch_size=64, seed=42, num_workers=8):
    """获取数据加载器"""
    set_seed(seed)

    # 获取所有slides
    all_slides = glob.glob(os.path.join(root_dir, "*"))
    # 过滤出文件夹
    all_slides = [slide for slide in all_slides if os.path.isdir(slide)]
    print(f"[INFO] Found {len(all_slides)} potential slide folders")
    statistics_slides(root_dir)

    # 创建数据集和数据加载器
    dataset = Camelyon16_Dataset(slides=all_slides)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloader


if __name__ == '__main__':
    # 简单测试代码
    root_dir = '/home/xiaoyuan/Data3/CAMELYON16/testing'
    batch_size = 64
    dataloader = get_dataloader(root_dir=root_dir, batch_size=batch_size, seed=42, num_workers=8)
    print(f"Dataset loaded with {len(dataloader.dataset)} patches")
    for batch_idx, (data, labels, index) in enumerate(dataloader):
        if batch_idx == 0:
            print(f"Batch {batch_idx}:")
            print(f"Data shape: {data.shape}")
            print(f"Labels: {labels}")
            print(f"Index: {index}")
        if batch_idx >= 5:
            break