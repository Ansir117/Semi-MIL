# test_camelyon16_official.py
# 测试CAMELYON16数据集在官方USB框架中的集成

import os
import sys
import torch
import numpy as np
from collections import Counter


def test_import():
    """测试模块导入"""
    print("=== 模块导入测试 ===")

    try:
        # 测试基础模块导入
        import semilearn
        print("✅ semilearn导入成功")

        # 测试CAMELYON16数据集导入
        from semilearn.datasets.cv_datasets.camelyon16 import get_camelyon16, analyze_camelyon16_dataset
        print("✅ CAMELYON16数据集模块导入成功")

        # 测试数据集注册
        from semilearn.datasets.cv_datasets import get_camelyon16 as registered_func
        print("✅ CAMELYON16数据集注册成功")

        return True

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请检查:")
        print("1. 是否正确创建了 camelyon16.py 文件")
        print("2. 是否正确修改了 __init__.py 文件")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False


def test_data_structure():
    """测试数据结构"""
    print("\n=== 数据结构测试 ===")

    data_dir = '/home/xiaoyuan/Data3/CAMELYON16'

    if not os.path.exists(data_dir):
        print(f"⚠️  数据目录不存在: {data_dir}")
        print("请更新测试脚本中的data_dir路径")
        return False

    neg_dir = os.path.join(data_dir, "Neg_Slide")
    pos_dir = os.path.join(data_dir, "Pos_Slide")

    if not os.path.exists(neg_dir):
        print(f"❌ 阴性数据目录不存在: {neg_dir}")
        return False

    if not os.path.exists(pos_dir):
        print(f"❌ 阳性数据目录不存在: {pos_dir}")
        return False

    print(f"✅ 数据目录结构正确")

    # 分析数据集
    try:
        from semilearn.datasets.cv_datasets.camelyon16 import analyze_camelyon16_dataset
        stats = analyze_camelyon16_dataset(data_dir)
        return True
    except Exception as e:
        print(f"❌ 数据分析失败: {e}")
        return False


def test_dataset_creation():
    """测试数据集创建"""
    print("\n=== 数据集创建测试 ===")

    try:
        # 模拟官方框架的args参数
        class MockArgs:
            def __init__(self):
                self.data_dir = '/home/xiaoyuan/Data3/CAMELYON16'
                self.img_size = 224
                self.seed = 0
                self.crop_ratio = 0.875
                self.ulb_num_labels = None
                self.lb_imb_ratio = 1
                self.ulb_imb_ratio = 1

        args = MockArgs()

        if not os.path.exists(args.data_dir):
            print(f"⚠️  跳过数据集创建测试，数据目录不存在: {args.data_dir}")
            return True

        # 导入数据集函数
        from semilearn.datasets.cv_datasets.camelyon16 import get_camelyon16

        # 创建数据集
        print("📝 创建数据集...")
        lb_dset, ulb_dset, eval_dset = get_camelyon16(
            args, 'instant', 'camelyon16', 500, 2, args.data_dir, True
        )

        print(f"✅ 数据集创建成功:")
        print(f"   有标签数据集: {len(lb_dset)} 样本")
        print(f"   无标签数据集: {len(ulb_dset)} 样本")
        print(f"   评估数据集: {len(eval_dset)} 样本")

        return True

    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """测试数据加载"""
    print("\n=== 数据加载测试 ===")

    try:
        # 创建简单的测试数据集
        class MockArgs:
            def __init__(self):
                self.data_dir = '/home/xiaoyuan/Data3/CAMELYON16'
                self.img_size = 224
                self.seed = 0
                self.crop_ratio = 0.875
                self.ulb_num_labels = None
                self.lb_imb_ratio = 1
                self.ulb_imb_ratio = 1

        args = MockArgs()

        if not os.path.exists(args.data_dir):
            print(f"⚠️  跳过数据加载测试，数据目录不存在")
            return True

        from semilearn.datasets.cv_datasets.camelyon16 import get_camelyon16

        # 创建小规模数据集用于测试
        lb_dset, ulb_dset, eval_dset = get_camelyon16(
            args, 'instant', 'camelyon16', 50, 2, args.data_dir, True  # 减少样本数用于测试
        )

        # 测试有标签数据加载
        if len(lb_dset) > 0:
            img, label = lb_dset[0]
            print(f"✅ 有标签数据加载成功:")
            print(f"   图像形状: {img.shape}")
            print(f"   图像类型: {type(img)}")
            print(f"   标签: {label}")
            print(f"   图像值范围: [{img.min():.3f}, {img.max():.3f}]")

            # 测试标签分布
            sample_size = min(100, len(lb_dset))
            labels = []
            for i in range(sample_size):
                _, label = lb_dset[i]
                labels.append(label)

            label_counts = Counter(labels)
            print(f"   标签分布 (样本{sample_size}): {dict(label_counts)}")

        # 测试无标签数据加载
        if len(ulb_dset) > 0:
            idx, img_w, img_s = ulb_dset[0]
            print(f"✅ 无标签数据加载成功:")
            print(f"   弱增强图像形状: {img_w.shape}")
            print(f"   强增强图像形状: {img_s.shape}")
            print(f"   样本索引: {idx}")

        # 测试测试集加载
        if len(eval_dset) > 0:
            img, label = eval_dset[0]
            print(f"✅ 测试数据加载成功:")
            print(f"   图像形状: {img.shape}")
            print(f"   标签: {label}")

        return True

    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_file():
    """测试配置文件"""
    print("\n=== 配置文件测试 ===")

    config_path = 'config/classic_cv/instant/instant_camelyon16.yaml'

    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        print("请确保已创建配置文件")
        return False

    try:
        import yaml

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 检查必要的配置项
        required_keys = [
            'algorithm', 'dataset', 'num_classes', 'num_labels',
            'data_dir', 'net', 'lr', 'batch_size'
        ]

        missing_keys = []
        for key in required_keys:
            if key not in config:
                missing_keys.append(key)

        if missing_keys:
            print(f"❌ 配置文件缺少必要参数: {missing_keys}")
            return False

        print(f"✅ 配置文件格式正确:")
        print(f"   算法: {config.get('algorithm')}")
        print(f"   数据集: {config.get('dataset')}")
        print(f"   网络: {config.get('net')}")
        print(f"   学习率: {config.get('lr')}")
        print(f"   批次大小: {config.get('batch_size')}")
        print(f"   数据路径: {config.get('data_dir')}")

        # 检查数据路径
        data_dir = config.get('data_dir')
        if data_dir and not os.path.exists(data_dir):
            print(f"⚠️  配置中的数据路径不存在: {data_dir}")
            print("请修改配置文件中的data_dir")

        return True

    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        return False


def test_training_readiness():
    """测试训练准备情况"""
    print("\n=== 训练准备测试 ===")

    try:
        # 检查train.py是否存在
        if not os.path.exists('train.py'):
            print("❌ train.py不存在")
            return False

        # 检查InstanT算法是否可用
        try:
            from semilearn.algorithms import get_algorithm
            print("✅ 算法模块导入成功")
        except ImportError:
            print("❌ 算法模块导入失败")
            return False

        # 检查GPU可用性
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.device_count()} 个GPU")
            print(f"   当前GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA不可用，将使用CPU训练")

        # 检查必要的依赖
        required_packages = ['torch', 'torchvision', 'numpy', 'PIL', 'yaml']
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package} 已安装")
            except ImportError:
                print(f"❌ {package} 未安装")
                return False

        print("✅ 训练环境准备就绪")
        return True

    except Exception as e:
        print(f"❌ 训练准备测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🧪 CAMELYON16官方框架集成测试")
    print("=" * 50)

    tests = [
        ("模块导入", test_import),
        ("数据结构", test_data_structure),
        ("数据集创建", test_dataset_creation),
        ("数据加载", test_data_loading),
        ("配置文件", test_config_file),
        ("训练准备", test_training_readiness),
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 执行 {test_name} 测试...")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")

    # 测试总结
    print("\n" + "=" * 50)
    print(f"🏁 测试完成")
    print(f"📊 通过率: {passed_tests}/{total_tests} ({passed_tests / total_tests * 100:.1f}%)")

    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！")
        print("\n🚀 现在可以开始训练:")
        print("   cd /path/to/your/USB_framework")
        print("   python train.py --c config/classic_cv/instant/instant_camelyon16.yaml")
        print("\n📊 监控训练:")
        print("   tensorboard --logdir saved_models/")
        return True
    elif passed_tests >= total_tests * 0.7:
        print("\n⚠️  大部分测试通过，可以尝试训练")
        print("如果训练时遇到问题，请检查失败的测试项")
        return True
    else:
        print("\n❌ 多个测试失败，请修复后再尝试训练")
        print("\n🔧 常见问题:")
        print("1. 检查数据路径是否正确")
        print("2. 确保所有文件都已正确创建")
        print("3. 验证 __init__.py 文件是否正确修改")
        print("4. 检查依赖包是否完整安装")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)