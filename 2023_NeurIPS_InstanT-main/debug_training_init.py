# debug_training_init.py
# 调试完整的训练初始化过程

import os
import sys
import yaml
import argparse


def debug_training_initialization():
    """调试完整的训练初始化过程"""
    print("🔍 调试训练初始化过程")
    print("=" * 60)

    try:
        # 1. 模拟命令行参数
        print("1️⃣ 设置命令行参数...")
        config_path = "config/classic_cv/instant/instant_camelyon16.yaml"

        # 模拟argparse的结果
        class MockArgs:
            def __init__(self):
                self.c = config_path
                self.gpu = 0
                self.world_size = 1
                self.rank = 0
                self.multiprocessing_distributed = False
                self.distributed = False

        args = MockArgs()
        print(f"   ✅ 参数设置完成: config={args.c}")

        # 2. 加载配置文件
        print("\n2️⃣ 加载配置文件...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 将配置添加到args
        for key, value in config.items():
            setattr(args, key, value)

        print(f"   ✅ 配置加载成功: {len(config)} 个参数")
        print(f"   📊 关键参数: algorithm={args.algorithm}, dataset={args.dataset}")

        # 3. 导入必要的模块
        print("\n3️⃣ 导入semilearn模块...")
        sys.path.append('.')

        from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm
        print("   ✅ semilearn模块导入成功")

        # 4. 测试数据集获取
        print("\n4️⃣ 测试get_dataset函数...")
        try:
            # 首先检查get_dataset的函数签名
            import inspect
            sig = inspect.signature(get_dataset)
            print(f"   get_dataset函数签名: {sig}")

            # 使用正确的参数调用
            dataset_dict = get_dataset(args, args.algorithm, args.dataset, args.num_labels, args.num_classes)

            if dataset_dict is None:
                print("   ❌ get_dataset返回None")
                return False

            print(f"   ✅ get_dataset成功: {type(dataset_dict)}")

            # 检查返回的数据集
            if isinstance(dataset_dict, dict):
                print("   📋 数据集字典内容:")
                for key, dataset in dataset_dict.items():
                    if dataset is not None:
                        print(f"     {key}: {len(dataset)} 样本")
                    else:
                        print(f"     {key}: None")
            elif isinstance(dataset_dict, (list, tuple)):
                print(f"   📋 数据集元组/列表: {len(dataset_dict)} 个元素")
                for i, dataset in enumerate(dataset_dict):
                    if dataset is not None:
                        print(f"     [{i}]: {len(dataset)} 样本")
                    else:
                        print(f"     [{i}]: None")
            else:
                print(f"   ⚠️  未知的数据集格式: {type(dataset_dict)}")

        except Exception as e:
            print(f"   ❌ get_dataset失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 5. 测试DataLoader创建
        print("\n5️⃣ 测试get_data_loader函数...")
        try:
            # 检查get_data_loader函数期望的输入格式
            import inspect
            sig = inspect.signature(get_data_loader)
            print(f"   get_data_loader签名: {sig}")

            # 检查dataset_dict的类型和内容
            print(f"   dataset_dict类型: {type(dataset_dict)}")

            if isinstance(dataset_dict, dict):
                print("   📋 dataset_dict是字典格式，需要分别创建DataLoader")

                # 为每个数据集分别创建DataLoader
                loader_dict = {}

                # 创建train_lb DataLoader
                if 'train_lb' in dataset_dict and dataset_dict['train_lb'] is not None:
                    print("   创建train_lb DataLoader...")
                    train_lb_loader = get_data_loader(
                        args=args,
                        dset=dataset_dict['train_lb'],
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        drop_last=True
                    )
                    loader_dict['train_lb'] = train_lb_loader
                    print(f"     ✅ train_lb DataLoader创建成功")

                # 创建train_ulb DataLoader
                if 'train_ulb' in dataset_dict and dataset_dict['train_ulb'] is not None:
                    print("   创建train_ulb DataLoader...")
                    train_ulb_loader = get_data_loader(
                        args=args,
                        dset=dataset_dict['train_ulb'],
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        drop_last=True
                    )
                    loader_dict['train_ulb'] = train_ulb_loader
                    print(f"     ✅ train_ulb DataLoader创建成功")

                # 创建eval DataLoader
                if 'eval' in dataset_dict and dataset_dict['eval'] is not None:
                    print("   创建eval DataLoader...")
                    eval_loader = get_data_loader(
                        args=args,
                        dset=dataset_dict['eval'],
                        batch_size=args.eval_batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        drop_last=False
                    )
                    loader_dict['eval'] = eval_loader
                    print(f"     ✅ eval DataLoader创建成功")

                # 创建test DataLoader（如果存在）
                if 'test' in dataset_dict and dataset_dict['test'] is not None:
                    print("   创建test DataLoader...")
                    test_loader = get_data_loader(
                        args=args,
                        dset=dataset_dict['test'],
                        batch_size=args.eval_batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        drop_last=False
                    )
                    loader_dict['test'] = test_loader
                    print(f"     ✅ test DataLoader创建成功")

            elif isinstance(dataset_dict, (list, tuple)):
                print("   📋 dataset_dict是元组/列表格式")
                # 如果是元组格式：(train_lb, train_ulb, eval_dset)
                train_lb, train_ulb, eval_dset = dataset_dict

                loader_dict = {}

                if train_lb is not None:
                    loader_dict['train_lb'] = get_data_loader(
                        args=args, dset=train_lb, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers, drop_last=True
                    )

                if train_ulb is not None:
                    loader_dict['train_ulb'] = get_data_loader(
                        args=args, dset=train_ulb, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers, drop_last=True
                    )

                if eval_dset is not None:
                    loader_dict['eval'] = get_data_loader(
                        args=args, dset=eval_dset, batch_size=args.eval_batch_size,
                        shuffle=False, num_workers=args.num_workers, drop_last=False
                    )

            else:
                print(f"   ❌ 未知的dataset格式: {type(dataset_dict)}")
                return False

            if not loader_dict:
                print("   ❌ loader_dict为空")
                return False

            print(f"   ✅ get_data_loader成功: {type(loader_dict)}")

            # 检查loader_dict内容
            print("   📋 DataLoader字典内容:")
            for key, loader in loader_dict.items():
                if loader is not None:
                    print(f"     {key}: {type(loader)}")
                else:
                    print(f"     {key}: None")

        except Exception as e:
            print(f"   ❌ get_data_loader失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 6. 测试网络构建
        print("\n6️⃣ 测试网络构建...")
        try:
            net_builder = get_net_builder(args.net, args.net_from_name)
            print(f"   ✅ 网络构建器创建成功: {type(net_builder)}")

        except Exception as e:
            print(f"   ❌ 网络构建失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 7. 测试算法初始化
        print("\n7️⃣ 测试算法初始化...")
        try:
            algorithm = get_algorithm(args, net_builder, tb_log=None, logger=None)
            print(f"   ✅ 算法创建成功: {type(algorithm)}")

            # 检查算法的loader_dict
            if hasattr(algorithm, 'loader_dict'):
                print(f"   📋 算法的loader_dict: {type(algorithm.loader_dict)}")
                if algorithm.loader_dict is None:
                    print("   ❌ 算法的loader_dict是None")

                    # 尝试手动设置
                    print("   🔧 尝试手动设置loader_dict...")
                    algorithm.loader_dict = loader_dict
                    print(f"   ✅ 手动设置后: {type(algorithm.loader_dict)}")
                else:
                    print("   ✅ 算法的loader_dict不为None")
            else:
                print("   ❌ 算法没有loader_dict属性")

        except Exception as e:
            print(f"   ❌ 算法初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 8. 测试训练开始
        print("\n8️⃣ 测试训练初始化...")
        try:
            # 检查train方法
            if hasattr(algorithm, 'train') and callable(algorithm.train):
                print("   ✅ 算法有train方法")

                # 检查loader_dict是否正确设置
                if algorithm.loader_dict is not None:
                    print("   ✅ loader_dict已设置")

                    # 检查必需的键
                    required_keys = ['train_lb', 'train_ulb']
                    for key in required_keys:
                        if key in algorithm.loader_dict:
                            if algorithm.loader_dict[key] is not None:
                                print(f"   ✅ {key}: 存在且不为None")
                            else:
                                print(f"   ❌ {key}: 存在但为None")
                        else:
                            print(f"   ❌ {key}: 不存在")
                else:
                    print("   ❌ loader_dict为None")
                    return False
            else:
                print("   ❌ 算法没有train方法")
                return False

        except Exception as e:
            print(f"   ❌ 训练初始化检查失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        print("\n🎉 所有初始化步骤检查完成!")
        print("💡 如果所有步骤都成功，训练应该可以开始")

        return True

    except Exception as e:
        print(f"❌ 调试过程出现异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_semilearn_imports():
    """检查semilearn的导入和版本"""
    print("\n🔍 检查semilearn模块")
    print("-" * 40)

    try:
        sys.path.append('.')

        # 检查各个模块的导入
        modules_to_check = [
            'semilearn.datasets',
            'semilearn.algorithms',
            'semilearn.core',
            'semilearn.nets'
        ]

        for module_name in modules_to_check:
            try:
                __import__(module_name)
                print(f"✅ {module_name}")
            except Exception as e:
                print(f"❌ {module_name}: {e}")

        # 检查主要函数
        from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm
        print("✅ 主要函数导入成功")

    except Exception as e:
        print(f"❌ semilearn检查失败: {e}")


if __name__ == "__main__":
    check_semilearn_imports()
    success = debug_training_initialization()

    if success:
        print("\n🚀 初始化调试成功，可以尝试训练")
    else:
        print("\n❌ 发现问题，需要修复后再训练")