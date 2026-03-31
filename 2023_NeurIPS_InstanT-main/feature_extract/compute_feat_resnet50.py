import numpy as np
import torch
import torch.nn as nn
from dataset_camelyon16 import Camelyon16_Dataset, get_dataloader
import os
import time
import h5py
from PIL import Image
import sys
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.models as models
import glob


class ResNet50FeatureExtractor(nn.Module):
    """ResNet50特征提取器，去除最后的分类层"""

    def __init__(self, pretrained=True):
        super(ResNet50FeatureExtractor, self).__init__()
        # 加载预训练的ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        # 去除最后的全连接层，保留特征提取部分
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平为 [batch_size, 2048]
        return x


def verify_model_files(model_dir):
    """验证模型文件是否存在"""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录 {model_dir} 不存在")

    # 查找可能的模型文件
    possible_files = []
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith(('.pth', '.pt', '.bin', '.ckpt')):
                possible_files.append(os.path.join(root, file))

    if not possible_files:
        raise FileNotFoundError(f"在目录 {model_dir} 中未找到模型文件 (.pth, .pt, .bin, .ckpt)")

    print(f"找到以下模型文件:")
    for file in possible_files:
        print(f"  - {file}")

    return possible_files


def load_resnet50_model(model_path, device="cuda"):
    """加载ResNet50模型"""
    print(f"正在加载ResNet50模型: {model_path}")

    # 创建特征提取器
    model = ResNet50FeatureExtractor(pretrained=False)

    try:
        # 加载保存的权重
        checkpoint = torch.load(model_path, map_location=device)

        # 根据不同的保存格式处理权重
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # 处理键名不匹配的问题
        model_dict = model.state_dict()
        filtered_dict = {}

        for k, v in state_dict.items():
            # 移除可能的前缀
            new_k = k
            if k.startswith('module.'):
                new_k = k[7:]  # 移除 'module.' 前缀
            elif k.startswith('encoder.'):
                new_k = k[8:]  # 移除 'encoder.' 前缀
            elif k.startswith('backbone.'):
                new_k = k[9:]  # 移除 'backbone.' 前缀

            # 检查是否需要映射到features模块
            if new_k in model_dict:
                filtered_dict[new_k] = v
            elif f'features.{new_k}' in model_dict:
                filtered_dict[f'features.{new_k}'] = v

        # 加载匹配的权重
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)

        print(f"成功加载权重，匹配的参数数量: {len(filtered_dict)}")

    except Exception as e:
        print(f"警告: 无法加载预训练权重: {e}")
        print("将使用随机初始化的权重")

    model = model.to(device)
    model.eval()
    return model


def compute_features(model, dataloader, device="cuda"):
    """计算特征"""
    L = len(dataloader.dataset)
    # ResNet50的特征维度是2048
    feat_dim = 2048
    feat_all = np.zeros([L, feat_dim], dtype=np.float32)
    error_count = 0
    success_count = 0

    # 创建进度条
    progress_bar = tqdm(dataloader, desc='计算特征')

    for data in progress_bar:
        try:
            images = data[0]  # 图像数据
            selected = data[2]  # 索引

            # 将图像移动到设备
            images = images.to(device)

            with torch.no_grad():
                # 使用ResNet50提取特征
                features = model(images)
                features = features.cpu().numpy()

                # 存储特征
                feat_all[selected] = features
                success_count += len(selected)

        except Exception as e:
            # 处理批次错误
            print(f"\nError processing batch: {e}")
            batch_size = len(data[2])
            error_count += batch_size

            # 使用零向量替代
            zero_features = np.zeros((batch_size, feat_dim), dtype=np.float32)
            try:
                feat_all[selected] = zero_features
            except:
                pass

        # 更新进度条描述，显示成功率
        if success_count + error_count > 0:
            progress_bar.set_postfix({
                "成功": success_count,
                "错误": error_count,
                "成功率": f"{(success_count / (success_count + error_count) * 100):.2f}%"
            })

    if success_count + error_count > 0:
        print(
            f"\n特征提取完成! 成功: {success_count}, 错误: {error_count}, 成功率: {(success_count / (success_count + error_count) * 100):.2f}%")
    else:
        print("\n特征提取完成，但未成功处理任何图像")

    return feat_all


def main():
    # 设置路径 - Camelyon16数据集路径
    input_dir = '/home/xiaoyuan/Data3/CAMELYON16/testing'  # Camelyon16训练数据目录
    output_dir = "/home/xiaoyuan/Data3/CAMELYON16/testing_feature"  # 输出目录
    model_dir = "/home/xiaoyuan/Semi-MIL/2023_NeurIPS_InstanT-main/saved_models/camelyon16_instant_cv_fold_5"  # ResNet50 模型目录

    os.makedirs(output_dir, exist_ok=True)

    # 记录数据集信息
    print(f"记录数据集信息到 {output_dir}/dataset_info.txt")
    with open(os.path.join(output_dir, 'dataset_info.txt'), 'w') as f:
        f.write(f"Dataset: CAMELYON16 Testing Set\n")
        f.write(f"Data source: {input_dir}\n")
        f.write(f"Processing time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: ResNet50 (directory: {model_dir})\n")
        f.write("Class mapping:\n")
        f.write("Class 0: Negative\n")
        f.write("Class 1: Positive\n")
        f.write("Patch size: 224x224 pixels\n")
        f.write("Slide naming format: prefix_number_label (e.g., test_001_1, tumor_001_0)\n")

    # 检查模型文件是否存在
    try:
        model_files = verify_model_files(model_dir)

        # 优先选择 model_best.pth，如果不存在则选择 latest_model.pth
        model_path = None
        for file_path in model_files:
            if 'model_best.pth' in file_path:
                model_path = file_path
                break

        if model_path is None:
            for file_path in model_files:
                if 'latest_model.pth' in file_path:
                    model_path = file_path
                    break

        if model_path is None:
            model_path = model_files[0]  # 如果都没找到，使用第一个文件

        print(f"使用模型文件: {model_path}")
    except Exception as e:
        print(f"模型文件验证失败: {e}")
        print("将使用随机初始化的ResNet50模型")
        model_path = None

    try:
        # 设置设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")

        # 加载ResNet50模型
        if model_path:
            model = load_resnet50_model(model_path, device)
        else:
            print("创建随机初始化的ResNet50模型...")
            model = ResNet50FeatureExtractor(pretrained=True)  # 使用ImageNet预训练权重
            model = model.to(device)
            model.eval()

        print("ResNet50模型准备完成")

        # 获取数据
        print("准备数据...")
        print(f"使用数据路径: {input_dir}")

        # 检查路径是否存在
        if not os.path.exists(input_dir):
            print(f"错误: 路径 {input_dir} 不存在，请检查路径设置")
            alternative_paths = [
                '/home/xiaoyuan/Data/CAMELYON16/patches_byDSMIL_224x224_10x/testing',
                '/data/CAMELYON16/patches_byDSMIL_224x224_10x/testing',
                './CAMELYON16/patches_byDSMIL_224x224_10x/testing'
            ]
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    print(f"找到替代路径: {alt_path}")
                    input_dir = alt_path
                    break
            else:
                raise FileNotFoundError(f"无法找到数据目录，请检查路径设置")

        # 列出一些示例文件夹和文件，帮助调试
        print("\n示例文件夹内容:")
        try:
            folders = os.listdir(input_dir)[:5]  # 获取前5个文件夹
            print(f"前5个文件夹: {folders}")

            if folders:
                first_folder = os.path.join(input_dir, folders[0])
                print(f"\n第一个文件夹 ({folders[0]}) 内容:")
                files = os.listdir(first_folder)[:5]  # 获取前5个文件
                print(f"前5个文件: {files}")
        except Exception as e:
            print(f"列出文件时出错: {e}")

        # 获取数据加载器
        dataloader = get_dataloader(
            root_dir=input_dir,
            batch_size=64,  # 根据GPU内存调整
            seed=42
        )
        print(f"数据加载器准备完成，共有{len(dataloader.dataset)}个patch")

        # 提取特征
        print("\n开始提取特征...")
        feat_all = compute_features(model, dataloader, device)

        # 保存结果
        print("\n保存特征和相关信息...")

        # 保存特征到h5文件
        h5f = h5py.File(os.path.join(output_dir, 'patch_feat.h5'), 'w')
        h5f.create_dataset('dataset_1', data=feat_all)
        h5f.close()

        # 保存其他信息
        np.save(os.path.join(output_dir, "feats.npy"), feat_all)
        np.save(os.path.join(output_dir, "patch_label.npy"),
                dataloader.dataset.patch_label)
        np.save(os.path.join(output_dir, "patch_pos_x.npy"),
                dataloader.dataset.patch_pos_x)
        np.save(os.path.join(output_dir, "patch_pos_y.npy"),
                dataloader.dataset.patch_pos_y)
        np.save(os.path.join(output_dir, "patch_corresponding_slide_label.npy"),
                dataloader.dataset.patch_corresponding_slide_label)
        np.save(os.path.join(output_dir, "patch_corresponding_slide_index.npy"),
                dataloader.dataset.patch_corresponding_slide_index)
        np.save(os.path.join(output_dir, "patch_corresponding_slide_name.npy"),
                dataloader.dataset.patch_corresponding_slide_name)

        # 保存特征维度信息
        with open(os.path.join(output_dir, 'feature_info.txt'), 'w') as f:
            f.write(f"Dataset: CAMELYON16 Testing Set\n")
            f.write(f"Model: ResNet50\n")
            f.write(f"特征维度: {feat_all.shape[1]}\n")
            f.write(f"处理的patch总数: {len(feat_all)}\n")
            labels = dataloader.dataset.patch_corresponding_slide_label
            f.write(f"类别分布: {np.bincount(labels)} (0:negative, 1:positive)\n")
            f.write(f"Patch size: 224x224 pixels\n")
            f.write("Slide naming format: prefix_number_label (e.g., test_001_1, tumor_001_0)\n")
            if model_path:
                f.write(f"Model path: {model_path}\n")
            else:
                f.write("Model: ImageNet pretrained ResNet50\n")

        print(f"\n特征提取完成!")
        print(f"结果保存至: {output_dir}")
        print(f"处理的patch总数: {len(feat_all)}")
        print(f"特征维度: {feat_all.shape[1]}")

        # 打印标签分布
        labels = dataloader.dataset.patch_corresponding_slide_label
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("\n标签分布:")
        for label, count in zip(unique_labels, counts):
            label_name = "negative" if label == 0 else "positive"
            print(f"类别 {label} ({label_name}): {count} patches ({count / len(labels) * 100:.2f}%)")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()