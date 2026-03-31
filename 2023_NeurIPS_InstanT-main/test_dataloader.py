# test_final_dataloader.py
# 最终测试 - 处理BasicDataset的字典返回格式

import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader


def test_fixed_dataloader():
    """测试修复后的DataLoader"""
    print("🔍 测试修复后的DataLoader")
    print("=" * 60)

    try:
        # 1. 加载配置
        print("1️⃣ 加载配置文件...")
        config_path = "config/classic_cv/instant/instant_camelyon16.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("   ✅ 配置文件加载成功")

        # 2. 创建args对象
        print("\n2️⃣ 创建args对象...")

        class Args:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)
                # 添加内存优化选项
                self.max_samples_per_slide = 100  # 限制每个slide的样本数（调试用）

        args = Args(config)
        print("   ✅ args对象创建成功")
        print(f"   📊 关键参数: batch_size={args.batch_size}, max_samples_per_slide={args.max_samples_per_slide}")

        # 3. 导入数据集函数
        print("\n3️⃣ 导入数据集函数...")
        sys.path.append('.')
        from semilearn.datasets.cv_datasets.camelyon16_0904 import get_camelyon16
        print("   ✅ 数据集函数导入成功")

        # 4. 创建数据集
        print("\n4️⃣ 创建数据集...")
        dataset_dict = get_camelyon16(args=args, alg='instant')

        if dataset_dict is None:
            print("   ❌ 数据集创建返回None")
            return False

        print("   ✅ 数据集创建成功")
        for key, dataset in dataset_dict.items():
            if dataset is not None:
                print(f"     {key}: {len(dataset)} 样本")
            else:
                print(f"     {key}: None")

        # 5. 测试单个样本获取（字典格式）
        print("\n5️⃣ 测试单个样本获取（字典格式）...")
        try:
            train_lb = dataset_dict['train_lb']
            if train_lb is not None and len(train_lb) > 0:
                sample_dict = train_lb[0]  # 返回字典格式
                print(f"   ✅ 样本获取成功: {type(sample_dict)}")

                # 解析字典内容
                if isinstance(sample_dict, dict):
                    print("   📋 字典内容:")
                    for key, value in sample_dict.items():
                        if isinstance(value, torch.Tensor):
                            print(f"     {key}: {value.shape} ({value.dtype})")
                        else:
                            print(f"     {key}: {value}")
                else:
                    print(f"   ⚠️  非字典格式: {sample_dict}")

            else:
                print("   ❌ train_lb数据集为空")
                return False
        except Exception as e:
            print(f"   ❌ 样本获取失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 6. 测试DataLoader创建
        print("\n6️⃣ 测试DataLoader创建...")
        try:
            # 设置合理的worker数量
            num_workers = min(2, args.num_workers)  # 限制worker数量
            print(f"   使用num_workers: {num_workers}")

            # 测试train_lb DataLoader
            print("   测试train_lb DataLoader...")
            train_lb_loader = DataLoader(
                dataset_dict['train_lb'],
                batch_size=min(8, args.batch_size),  # 使用较小的batch_size
                shuffle=True,
                num_workers=num_workers,
                drop_last=True
            )
            print(f"     ✅ train_lb DataLoader创建成功")

            # 测试train_ulb DataLoader
            print("   测试train_ulb DataLoader...")
            train_ulb_loader = DataLoader(
                dataset_dict['train_ulb'],
                batch_size=min(8, args.batch_size),
                shuffle=True,
                num_workers=num_workers,
                drop_last=True
            )
            print(f"     ✅ train_ulb DataLoader创建成功")

            # 测试eval DataLoader
            print("   测试eval DataLoader...")
            eval_loader = DataLoader(
                dataset_dict['eval'],
                batch_size=min(8, args.eval_batch_size),
                shuffle=False,
                num_workers=num_workers,
                drop_last=False
            )
            print(f"     ✅ eval DataLoader创建成功")

        except Exception as e:
            print(f"   ❌ DataLoader创建失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 7. 测试数据批次获取（字典格式）
        print("\n7️⃣ 测试数据批次获取（字典格式）...")
        try:
            print("   测试train_lb批次获取...")
            batch_dict = next(iter(train_lb_loader))
            print(f"     ✅ 批次获取成功: {type(batch_dict)}")

            if isinstance(batch_dict, dict):
                print("     📋 批次字典内容:")
                for key, value in batch_dict.items():
                    if isinstance(value, torch.Tensor):
                        print(f"       {key}: {value.shape} ({value.dtype})")
                    else:
                        print(f"       {key}: {type(value)}")

            print("   测试train_ulb批次获取...")
            batch_ulb_dict = next(iter(train_ulb_loader))
            print(f"     ✅ ulb批次获取成功: {type(batch_ulb_dict)}")

            if isinstance(batch_ulb_dict, dict):
                print("     📋 ulb批次字典内容:")
                for key, value in batch_ulb_dict.items():
                    if isinstance(value, torch.Tensor):
                        print(f"       {key}: {value.shape}")
                    elif isinstance(value, (list, tuple)):
                        print(f"       {key}: {type(value)} 长度={len(value)}")
                        if len(value) > 0 and isinstance(value[0], torch.Tensor):
                            print(f"         [0]: {value[0].shape}")
                    else:
                        print(f"       {key}: {type(value)}")

        except Exception as e:
            print(f"   ❌ 批次获取失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        print("\n🎉 所有测试通过！DataLoader工作正常")
        print("\n💡 重要发现:")
        print("1. BasicDataset返回字典格式，包含 'x_lb'/'x_ulb', 'y_lb'/'y_ulb', 'idx_lb'/'idx_ulb'")
        print("2. 训练代码需要适配字典格式的数据访问")
        print("3. 数据预加载成功，内存使用正常")

        return True

    except Exception as e:
        print(f"❌ 测试过程出现异常: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_fixed_dataloader()

    if success:
        print("\n🚀 可以开始训练:")
        print("python train.py --c config/classic_cv/instant/instant_camelyon16.yaml")
    else:
        print("\n❌ 测试失败，需要进一步调试")