# detailed_basicdataset_test.py
# 详细测试BasicDataset的行为

import os
import sys
from PIL import Image
import torch
from torchvision import transforms


def analyze_basicdataset():
    """详细分析BasicDataset的行为"""
    print("🔍 详细分析BasicDataset")
    print("=" * 60)

    sys.path.append('.')
    from semilearn.datasets.cv_datasets.datasetbase import BasicDataset

    # 创建测试变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 找一个测试图像
    data_dir = "/home/xiaoyuan/Data3/CAMELYON16"
    test_image_path = None

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.jpg', '.png')):
                test_image_path = os.path.join(root, file)
                break
        if test_image_path:
            break

    if not test_image_path:
        print("❌ 找不到测试图像文件")
        return

    print(f"📸 使用测试图像: {test_image_path}")

    # 预加载图像
    img = Image.open(test_image_path).convert('RGB')
    print(f"✅ 图像加载成功: {img.size}")

    # 创建测试数据
    data = [img, img]
    targets = [0, 1]

    print(f"\n📊 输入数据信息:")
    print(f"  data类型: {type(data)}")
    print(f"  data长度: {len(data)}")
    print(f"  data[0]类型: {type(data[0])}")
    print(f"  targets类型: {type(targets)}")
    print(f"  targets长度: {len(targets)}")
    print(f"  targets内容: {targets}")

    try:
        # 创建BasicDataset
        print(f"\n🏗️ 创建BasicDataset...")
        dataset = BasicDataset(
            alg='test',
            data=data,
            targets=targets,
            num_classes=2,
            transform=transform,
            is_ulb=False,
            strong_transform=None,
            return_idx=False
        )

        print(f"✅ BasicDataset创建成功")

        # 检查dataset属性
        print(f"\n🔍 检查dataset属性:")
        print(f"  len(dataset): {len(dataset)}")
        print(f"  dataset.data类型: {type(dataset.data)}")
        print(f"  dataset.targets类型: {type(dataset.targets)}")

        # 检查dataset的关键属性
        attrs = ['data', 'targets', 'num_classes', 'transform', 'is_ulb']
        for attr in attrs:
            if hasattr(dataset, attr):
                value = getattr(dataset, attr)
                if attr == 'data':
                    data_len = len(value) if hasattr(value, "__len__") else "?"
                    value_display = f'[...{data_len}...]'
                else:
                    value_display = str(value)
                print(f"  dataset.{attr}: {type(value)} = {value_display}")
            else:
                print(f"  dataset.{attr}: 不存在")

        # 尝试访问索引0
        print(f"\n🎯 尝试访问dataset[0]...")
        try:
            sample = dataset[0]
            print(f"✅ 成功获取sample: {type(sample)}")
            if isinstance(sample, (tuple, list)):
                print(f"  sample长度: {len(sample)}")
                for i, item in enumerate(sample):
                    print(f"  sample[{i}]: {type(item)}")
                    if hasattr(item, 'shape'):
                        print(f"    形状: {item.shape}")
                    else:
                        print(f"    值: {item}")
            else:
                print(f"  sample内容: {sample}")

        except Exception as e:
            print(f"❌ 访问dataset[0]失败: {e}")
            print(f"错误类型: {type(e)}")
            import traceback
            traceback.print_exc()

        # 尝试访问dataset.__getitem__的内部逻辑
        print(f"\n🔧 尝试调试__getitem__内部...")
        try:
            # 手动模拟__getitem__(0)
            idx = 0
            print(f"  索引: {idx}")

            # 检查数据访问
            if hasattr(dataset, 'data') and idx < len(dataset.data):
                data_item = dataset.data[idx]
                print(f"  data[{idx}]: {type(data_item)}")
            else:
                print(f"  ❌ 无法访问data[{idx}]")

            if hasattr(dataset, 'targets') and idx < len(dataset.targets):
                target_item = dataset.targets[idx]
                print(f"  targets[{idx}]: {type(target_item)} = {target_item}")
            else:
                print(f"  ❌ 无法访问targets[{idx}]")

        except Exception as e:
            print(f"❌ 内部调试失败: {e}")

    except Exception as e:
        print(f"❌ BasicDataset创建失败: {e}")
        import traceback
        traceback.print_exc()


def check_cifar_implementation():
    """检查CIFAR实现方式"""
    print(f"\n🔍 检查CIFAR实现方式")
    print("-" * 40)

    cifar_file = "semilearn/datasets/cv_datasets/cifar.py"
    if os.path.exists(cifar_file):
        print(f"📄 读取 {cifar_file}")
        try:
            with open(cifar_file, 'r') as f:
                content = f.read()

            # 查找BasicDataset的使用
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'BasicDataset(' in line:
                    print(f"\nCIFAR中BasicDataset使用 (第{i + 1}行):")
                    # 显示上下文
                    start = max(0, i - 5)
                    end = min(len(lines), i + 10)
                    for j in range(start, end):
                        marker = ">>> " if j == i else "    "
                        print(f"{marker}{j + 1:3d}: {lines[j]}")
                    break

        except Exception as e:
            print(f"读取CIFAR文件失败: {e}")
    else:
        print(f"❌ CIFAR文件不存在: {cifar_file}")


if __name__ == "__main__":
    analyze_basicdataset()
    check_cifar_implementation()