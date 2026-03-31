# debug_data_structure.py
# 检查CAMELYON16数据结构和数据加载器问题

import os
import sys
import yaml


def check_data_directory():
    """检查数据目录结构"""
    print("🔍 检查CAMELYON16数据目录结构")
    print("-" * 50)

    data_dir = "/home/xiaoyuan/Data3/CAMELYON16"

    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return False

    print(f"✅ 数据目录存在: {data_dir}")

    # 列出目录内容
    print("\n📁 目录内容:")
    try:
        items = os.listdir(data_dir)
        for item in sorted(items):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                subdir_count = len(os.listdir(item_path)) if os.path.isdir(item_path) else 0
                print(f"  📁 {item}/ ({subdir_count} items)")
            else:
                size = os.path.getsize(item_path) / (1024 * 1024)  # MB
                print(f"  📄 {item} ({size:.1f} MB)")
    except Exception as e:
        print(f"❌ 读取目录失败: {e}")
        return False

    return True


def check_expected_structure():
    """检查期望的数据结构"""
    print("\n🎯 检查期望的数据结构")
    print("-" * 50)

    data_dir = "/home/xiaoyuan/Data3/CAMELYON16"
    expected_structure = [
        "train",
        "test",
        "valid",
        # 或者可能是
        "training",
        "testing",
        "validation"
    ]

    found_structure = []

    for folder in expected_structure:
        folder_path = os.path.join(data_dir, folder)
        if os.path.exists(folder_path):
            found_structure.append(folder)
            # 检查子目录
            try:
                subfolders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
                file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
                print(f"✅ {folder}/ - 子目录: {subfolders}, 文件数: {file_count}")
            except Exception as e:
                print(f"⚠️  {folder}/ - 读取失败: {e}")

    if not found_structure:
        print("❌ 未找到标准的训练/测试目录结构")
        return False

    return True


def check_camelyon16_dataset():
    """检查camelyon16数据集定义"""
    print("\n🧪 检查camelyon16数据集定义")
    print("-" * 50)

    try:
        sys.path.append('.')
        from semilearn.datasets.cv_datasets.camelyon16 import get_camelyon16
        print("✅ camelyon16数据集模块导入成功")

        # 尝试获取数据集
        try:
            # 读取配置
            config_path = "config/classic_cv/instant/instant_camelyon16.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            print(f"📋 配置信息:")
            print(f"  数据目录: {config.get('data_dir')}")
            print(f"  标记样本数: {config.get('num_labels')}")
            print(f"  批次大小: {config.get('batch_size')}")

            # 尝试创建数据集
            print("\n🔄 尝试创建数据集...")

            # 创建args对象
            class Args:
                def __init__(self, config_dict):
                    for key, value in config_dict.items():
                        setattr(self, key, value)

            args = Args(config)

            dataset_dict = get_camelyon16(
                args=args,
                alg='instant'  # 使用alg而不是algorithm
            )

            if dataset_dict is None:
                print("❌ 数据集创建返回None")
                return False

            print("✅ 数据集创建成功")

            # 检查数据集内容
            for key, dataset in dataset_dict.items():
                if dataset is not None:
                    print(f"  {key}: {len(dataset)} 样本")
                else:
                    print(f"  {key}: None")

            return True

        except Exception as e:
            print(f"❌ 数据集创建失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    except ImportError as e:
        print(f"❌ camelyon16模块导入失败: {e}")
        return False


def check_config_file():
    """检查配置文件"""
    print("\n⚙️  检查配置文件")
    print("-" * 50)

    config_path = "config/classic_cv/instant/instant_camelyon16.yaml"

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print("✅ 配置文件加载成功")

        # 关键配置检查
        critical_keys = ['data_dir', 'dataset', 'num_labels', 'batch_size']
        for key in critical_keys:
            value = config.get(key)
            print(f"  {key}: {value}")

            if key == 'data_dir' and not os.path.exists(value):
                print(f"    ❌ 数据目录不存在")

        return True

    except Exception as e:
        print(f"❌ 配置文件检查失败: {e}")
        return False


def main():
    """主检查函数"""
    print("🔍 CAMELYON16数据加载器调试")
    print("=" * 60)

    checks = [
        ("数据目录", check_data_directory),
        ("数据结构", check_expected_structure),
        ("配置文件", check_config_file),
        ("数据集定义", check_camelyon16_dataset)
    ]

    passed = 0

    for check_name, check_func in checks:
        print(f"\n{'=' * 20} {check_name} {'=' * 20}")
        try:
            if check_func():
                passed += 1
                print(f"✅ {check_name}检查通过")
            else:
                print(f"❌ {check_name}检查失败")
        except Exception as e:
            print(f"❌ {check_name}检查异常: {e}")

    print(f"\n📊 检查结果: {passed}/{len(checks)} 通过")

    if passed < len(checks):
        print("\n💡 调试建议:")
        print("1. 检查数据目录结构是否正确")
        print("2. 确认camelyon16.py文件实现正确")
        print("3. 验证配置文件中的路径")
        print("4. 检查数据集文件格式")


if __name__ == "__main__":
    main()