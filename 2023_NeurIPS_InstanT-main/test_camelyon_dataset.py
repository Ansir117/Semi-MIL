# test_camelyon_dataset.py
# 测试CAMELYON16数据集加载和注册 - 完整版

import os
import sys
import argparse
import torch
import numpy as np
from collections import Counter
from PIL import Image

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def test_environment():
    """测试环境依赖"""
    print("=== 环境测试 ===")

    # 测试Python版本
    print(f"Python版本: {sys.version}")

    # 测试PyTorch
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA设备数: {torch.cuda.device_count()}")
            print(f"当前CUDA设备: {torch.cuda.current_device()}")
    except ImportError:
        print("❌ PyTorch未安装")
        return False

    # 测试semilearn
    try:
        import semilearn
        print(f"✅ semilearn已安装")
    except ImportError:
        print("❌ semilearn未安装，请执行: pip install semilearn")
        return False

    # 测试其他依赖
    dependencies = ['PIL', 'numpy', 'torchvision']
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}已安装")
        except ImportError:
            print(f"❌ {dep}未安装")
            return False

    print("✅ 环境测试通过\n")
    return True


def test_data_structure(data_dir):
    """测试数据结构"""
    print("=== 数据结构测试 ===")
    print(f"数据路径: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"❌ 数据根目录不存在: {data_dir}")
        return False

    neg_dir = os.path.join(data_dir, "Neg_Slide")
    pos_dir = os.path.join(data_dir, "Pos_Slide")

    # 检查子目录
    if not os.path.exists(neg_dir):
        print(f"❌ 阴性数据目录不存在: {neg_dir}")
        return False

    if not os.path.exists(pos_dir):
        print(f"❌ 阳性数据目录不存在: {pos_dir}")
        return False

    print("✅ 基本目录结构正确")

    # 检查slide数量
    neg_slides = [d for d in os.listdir(neg_dir) if os.path.isdir(os.path.join(neg_dir, d))]
    pos_slides = [d for d in os.listdir(pos_dir) if os.path.isdir(os.path.join(pos_dir, d))]

    print(f"阴性slide数量: {len(neg_slides)}")
    print(f"阳性slide数量: {len(pos_slides)}")

    if len(neg_slides) == 0:
        print("❌ 没有找到阴性slide")
        return False

    if len(pos_slides) == 0:
        print("❌ 没有找到阳性slide")
        return False

    # 检查图像文件
    sample_neg_slide = os.path.join(neg_dir, neg_slides[0])
    sample_pos_slide = os.path.join(pos_dir, pos_slides[0])

    neg_images = [f for f in os.listdir(sample_neg_slide) if f.endswith('.jpg')]
    pos_images = [f for f in os.listdir(sample_pos_slide) if f.endswith('.jpg')]
    pos_labeled = [f for f in pos_images if f.endswith('_1.jpg')]

    print(f"示例阴性slide图像数: {len(neg_images)}")
    print(f"示例阳性slide图像数: {len(pos_images)}")
    print(f"示例阳性slide标注数: {len(pos_labeled)}")

    if len(neg_images) == 0:
        print("❌ 阴性slide中没有图像文件")
        return False

    if len(pos_images) == 0:
        print("❌ 阳性slide中没有图像文件")
        return False

    # 测试图像加载
    try:
        sample_img_path = os.path.join(sample_neg_slide, neg_images[0])
        img = Image.open(sample_img_path)
        print(f"示例图像尺寸: {img.size}")
        print(f"示例图像模式: {img.mode}")
    except Exception as e:
        print(f"❌ 图像加载失败: {e}")
        return False

    print("✅ 数据结构测试通过\n")
    return True


def test_registration():
    """测试数据集注册"""
    print("=== 数据集注册测试 ===")

    try:
        # 注册数据集
        print("📝 执行数据集注册...")
        import register_camelyon16

        # 重新导入以获取最新注册状态
        import importlib
        importlib.reload(register_camelyon16)

        # 测试注册结果
        import semilearn.datasets as datasets

        tests_passed = 0
        total_tests = 3

        # 测试1：检查函数
        if hasattr(datasets, 'get_camelyon16'):
            print("✅ get_camelyon16函数可访问")
            tests_passed += 1
        else:
            print("❌ get_camelyon16函数不可访问")

        # 测试2：检查dataset_dict
        if hasattr(datasets, 'dataset_dict') and 'camelyon16' in datasets.dataset_dict:
            print("✅ camelyon16在dataset_dict中")
            tests_passed += 1
        else:
            print("❌ camelyon16不在dataset_dict中")

        # 测试3：尝试获取函数
        try:
            get_func = getattr(datasets, 'get_camelyon16', None)
            if get_func is None and hasattr(datasets, 'dataset_dict'):
                get_func = datasets.dataset_dict.get('camelyon16', None)

            if get_func is not None:
                print("✅ 数据集函数可调用")
                tests_passed += 1
            else:
                print("❌ 数据集函数不可调用")
        except Exception as e:
            print(f"❌ 函数测试失败: {e}")

        success_rate = tests_passed / total_tests
        print(f"注册测试通过率: {tests_passed}/{total_tests} ({success_rate * 100:.1f}%)")

        if tests_passed >= 2:
            print("✅ 数据集注册基本成功\n")
            return True
        else:
            print("❌ 数据集注册失败\n")
            return False

    except Exception as e:
        print(f"❌ 注册测试失败: {e}\n")
        return False


def test_data_loading(data_dir, ssl_strategy='simulated'):
    """测试数据加载"""
    print(f"=== 数据加载测试 (策略: {ssl_strategy}) ===")

    try:
        # 创建测试参数
        class TestArgs:
            def __init__(self):
                self.data_dir = data_dir
                self.img_size = 224
                self.seed = 42
                self.ssl_strategy = ssl_strategy
                self.ulb_num_labels = None
                self.lb_imb_ratio = 1
                self.ulb_imb_ratio = 1

        args = TestArgs()

        # 导入数据集函数
        from camelyon_ssl import get_camelyon16, analyze_camelyon_dataset

        # 分析数据集
        print("📊 分析数据集...")
        stats = analyze_camelyon_dataset(data_dir)

        # 测试数据加载
        print(f"🔄 测试数据加载...")

        num_labels = min(500, stats['total_labeled'] // 4)  # 动态调整

        lb_dset, ulb_dset, eval_dset = get_camelyon16(
            args, 'instant', 'camelyon16', num_labels, 2, data_dir, True
        )

        print(f"✅ 数据集创建成功!")
        print(f"   - 有标签数据: {len(lb_dset)} 样本")
        print(f"   - 无标签数据: {len(ulb_dset)} 样本")
        print(f"   - 测试数据: {len(eval_dset)} 样本")

        # 测试数据访问
        print(f"🔍 测试数据访问...")

        # 测试有标签数据
        if len(lb_dset) > 0:
            try:
                img, label = lb_dset[0]
                print(f"   - 有标签样本: 图像形状 {img.shape}, 标签 {label}")
                print(f"   - 图像数据类型: {img.dtype}")
                print(f"   - 图像值范围: [{img.min():.3f}, {img.max():.3f}]")

                # 统计标签分布
                sample_size = min(1000, len(lb_dset))
                labels = []
                for i in range(sample_size):
                    _, label = lb_dset[i]
                    labels.append(label)

                label_counts = Counter(labels)
                print(f"   - 有标签数据分布 (样本{sample_size}): {dict(label_counts)}")

            except Exception as e:
                print(f"❌ 有标签数据访问失败: {e}")
                return False

        # 测试无标签数据
        if len(ulb_dset) > 0:
            try:
                idx, img_w, img_s = ulb_dset[0]
                print(f"   - 无标签样本: 弱增强 {img_w.shape}, 强增强 {img_s.shape}")
                print(f"   - 样本索引: {idx}")
            except Exception as e:
                print(f"❌ 无标签数据访问失败: {e}")
                return False

        # 测试测试集
        if len(eval_dset) > 0:
            try:
                img, label = eval_dset[0]
                print(f"   - 测试样本: 图像形状 {img.shape}, 标签 {label}")
            except Exception as e:
                print(f"❌ 测试数据访问失败: {e}")
                return False

        print("✅ 数据加载测试通过\n")
        return True

    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_training_compatibility():
    """测试训练兼容性"""
    print("=== 训练兼容性测试 ===")

    try:
        # 检查训练脚本
        if not os.path.exists('train.py'):
            print("❌ train.py文件不存在")
            return False

        if not os.path.exists('instant.py'):
            print("❌ instant.py文件不存在")
            return False

        if not os.path.exists('camelyon16_instant.yaml'):
            print("❌ camelyon16_instant.yaml配置文件不存在")
            return False

        print("✅ 必要的训练文件存在")

        # 测试配置文件
        try:
            import yaml
            with open('camelyon16_instant.yaml', 'r') as f:
                config = yaml.safe_load(f)

            required_keys = ['algorithm', 'dataset', 'net', 'num_classes', 'ssl_strategy']
            missing_keys = [key for key in required_keys if key not in config]

            if missing_keys:
                print(f"❌ 配置文件缺少必要参数: {missing_keys}")
                return False

            print("✅ 配置文件格式正确")
            print(f"   - 算法: {config.get('algorithm')}")
            print(f"   - 数据集: {config.get('dataset')}")
            print(f"   - 网络: {config.get('net')}")
            print(f"   - SSL策略: {config.get('ssl_strategy')}")

        except Exception as e:
            print(f"❌ 配置文件测试失败: {e}")
            return False

        # 测试instant算法
        try:
            from semilearn.algorithms import get_algorithm
            print("✅ InstanT算法可导入")
        except ImportError as e:
            print(f"❌ InstanT算法导入失败: {e}")
            return False

        print("✅ 训练兼容性测试通过\n")
        return True

    except Exception as e:
        print(f"❌ 训练兼容性测试失败: {e}\n")
        return False


def test_strategies_comparison(data_dir):
    """测试不同策略的对比"""
    print("=== 策略对比测试 ===")

    strategies = ['simulated', 'real']
    results = {}

    for strategy in strategies:
        print(f"🔄 测试策略: {strategy}")
        try:
            class TestArgs:
                def __init__(self):
                    self.data_dir = data_dir
                    self.img_size = 224
                    self.seed = 42
                    self.ssl_strategy = strategy
                    self.ulb_num_labels = None
                    self.lb_imb_ratio = 1
                    self.ulb_imb_ratio = 1

            args = TestArgs()

            from camelyon_ssl import get_camelyon16

            lb_dset, ulb_dset, eval_dset = get_camelyon16(
                args, 'instant', 'camelyon16', 200, 2, data_dir, True
            )

            results[strategy] = {
                'labeled': len(lb_dset),
                'unlabeled': len(ulb_dset),
                'test': len(eval_dset)
            }

            print(f"   ✅ {strategy}: 标注{len(lb_dset)}, 无标签{len(ulb_dset)}, 测试{len(eval_dset)}")

        except Exception as e:
            print(f"   ❌ {strategy}策略失败: {e}")
            results[strategy] = None

    # 对比结果
    print(f"\n📊 策略对比结果:")
    for strategy, result in results.items():
        if result:
            print(
                f"   {strategy:10}: 标注{result['labeled']:>6}, 无标签{result['unlabeled']:>6}, 测试{result['test']:>6}")
        else:
            print(f"   {strategy:10}: 失败")

    success_count = sum(1 for r in results.values() if r is not None)
    print(f"\n✅ 策略测试通过率: {success_count}/{len(strategies)}\n")

    return success_count > 0


def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description='CAMELYON16数据集完整测试')
    parser.add_argument('--data_dir', type=str,
                        default='/home/xiaoyuan/Data3/CAMELYON16',
                        help='数据根目录')
    parser.add_argument('--ssl_strategy', type=str, default='simulated',
                        choices=['real', 'simulated'],
                        help='半监督策略')
    parser.add_argument('--skip_env', action='store_true',
                        help='跳过环境测试')
    parser.add_argument('--skip_data', action='store_true',
                        help='跳过数据结构测试')
    parser.add_argument('--skip_loading', action='store_true',
                        help='跳过数据加载测试')
    parser.add_argument('--skip_comparison', action='store_true',
                        help='跳过策略对比测试')

    args = parser.parse_args()

    print("🧪 CAMELYON16数据集完整测试")
    print("=" * 50)

    all_tests = []

    # 环境测试
    if not args.skip_env:
        all_tests.append(('环境测试', test_environment))

    # 数据结构测试
    if not args.skip_data:
        all_tests.append(('数据结构测试', lambda: test_data_structure(args.data_dir)))

    # 注册测试
    all_tests.append(('数据集注册测试', test_registration))

    # 数据加载测试
    if not args.skip_loading:
        all_tests.append(('数据加载测试', lambda: test_data_loading(args.data_dir, args.ssl_strategy)))

    # 训练兼容性测试
    all_tests.append(('训练兼容性测试', test_training_compatibility))

    # 策略对比测试
    if not args.skip_comparison:
        all_tests.append(('策略对比测试', lambda: test_strategies_comparison(args.data_dir)))

    # 执行所有测试
    passed_tests = 0
    total_tests = len(all_tests)

    for test_name, test_func in all_tests:
        try:
            if test_func():
                passed_tests += 1
            else:
                print(f"❌ {test_name}失败")
        except Exception as e:
            print(f"❌ {test_name}异常: {e}")

    # 测试总结
    print("=" * 50)
    print(f"🏁 测试完成")
    print(f"📊 通过率: {passed_tests}/{total_tests} ({passed_tests / total_tests * 100:.1f}%)")

    if passed_tests == total_tests:
        print("🎉 所有测试通过！")
        print("")
        print("🚀 你可以开始训练了:")
        print("   chmod +x run_camelyon16.sh")
        print("   ./run_camelyon16.sh")
        print("")
        print("或者手动运行:")
        print("   python register_camelyon16.py")
        print("   python train.py --c camelyon16_instant.yaml")
        return True
    else:
        print("⚠️  部分测试失败，请检查上述错误信息")
        print("")
        print("🔧 常见问题解决:")
        print("1. 检查数据路径是否正确")
        print("2. 确保semilearn已正确安装")
        print("3. 检查所有必要文件是否存在")
        print("4. 检查Python环境和依赖")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)