# run_cross_validation.py
# CAMELYON16 InstanT算法5折交叉验证脚本

import os
import sys
import yaml
import argparse
import subprocess
import json
from datetime import datetime
import numpy as np

# 训练脚本名称，请根据实际情况修改
TRAIN_SCRIPT = "train.py"


def create_cv_config(base_config_path, cv_fold, cv_splits, save_dir):
    """
    为指定的交叉验证折创建配置文件

    Args:
        base_config_path: 基础配置文件路径
        cv_fold: 当前折数 (0-4)
        cv_splits: 交叉验证slide划分
        save_dir: 保存目录

    Returns:
        str: 新配置文件的路径
    """
    # 读取基础配置
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 修改配置参数
    config['data_seed'] = 100 + cv_fold * 10  # 为每折设置不同的种子
    config['seed'] = cv_fold  # 训练随机种子
    config['save_name'] = f"camelyon16_instant_cv_fold_{cv_fold + 1}"
    config['cv_fold'] = cv_fold  # 设置为0, 1, 2, 3, 4
    config['cv_splits'] = cv_splits if cv_splits else []

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 保存新配置文件
    cv_config_path = os.path.join(save_dir, f"instant_camelyon16_cv_fold_{cv_fold + 1}.yaml")
    with open(cv_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"  ✅ 创建配置文件: {cv_config_path}")
    print(f"     - data_seed: {config['data_seed']}")
    print(f"     - train_seed: {config['seed']}")
    print(f"     - save_name: {config['save_name']}")

    return cv_config_path


def create_cross_validation_splits(data_dir, num_folds=5, num_slides_per_fold=10):
    """
    预先创建交叉验证的slide划分
    调用数据集代码来获取slide信息
    """
    print(f"\n🔄 预先创建 {num_folds} 折交叉验证slide划分...")

    # 修正：导入原来的camelyon16模块，避免函数名冲突
    try:
        from camelyon16 import collect_camelyon16_paths_by_official_split
        from camelyon16 import create_cross_validation_splits as dataset_create_cv_splits

        # 收集数据路径以获取slide信息
        (train_labeled_paths, train_labeled_targets, train_unlabeled_paths,
         test_paths, test_targets) = collect_camelyon16_paths_by_official_split(data_dir)

        # 创建交叉验证划分
        cv_splits = dataset_create_cv_splits(
            train_labeled_paths, train_labeled_targets,
            num_folds=num_folds,
            num_slides_per_fold=num_slides_per_fold
        )

        return cv_splits

    except ImportError as e:
        print(f"❌ 无法导入数据集代码: {e}")
        print("  请确保 camelyon16.py 在Python路径中，并且包含交叉验证函数")

        # 备选方案：返回None，让每个训练折自己随机选择
        print("  将使用随机选择方案（每折独立选择slide）")
        return None
    except Exception as e:
        print(f"❌ 创建交叉验证划分失败: {e}")
        print(f"  错误详情: {str(e)}")
        return None


def run_single_fold(config_path, fold_num, log_dir):
    """
    运行单个交叉验证折

    Args:
        config_path: 配置文件路径
        fold_num: 折数
        log_dir: 日志保存目录

    Returns:
        dict: 训练结果
    """
    print(f"\n🚀 开始运行第 {fold_num + 1} 折训练...")
    print(f"  配置文件: {config_path}")

    # 创建日志文件
    log_file = os.path.join(log_dir, f"fold_{fold_num + 1}_training.log")
    error_log = os.path.join(log_dir, f"fold_{fold_num + 1}_error.log")

    # 构建训练命令 - 修正：使用正确的参数名 --c
    cmd = [
        "python", TRAIN_SCRIPT,
        "--c", config_path,  # 修正：原来是 --config
        "--gpu", "0"  # 根据需要调整GPU设置
    ]

    print(f"  执行命令: {' '.join(cmd)}")

    try:
        # 运行训练
        with open(log_file, 'w') as log_f, open(error_log, 'w') as err_f:
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=err_f,
                universal_newlines=True
            )

            # 等待训练完成
            return_code = process.wait()

            if return_code == 0:
                print(f"  ✅ 第 {fold_num + 1} 折训练完成")

                # 尝试解析结果
                result = parse_training_result(log_file, fold_num)
                return result

            else:
                print(f"  ❌ 第 {fold_num + 1} 折训练失败，返回码: {return_code}")
                print(f"  错误日志: {error_log}")
                return {"fold": fold_num + 1, "status": "failed", "return_code": return_code}

    except Exception as e:
        print(f"  ❌ 第 {fold_num + 1} 折训练异常: {e}")
        return {"fold": fold_num + 1, "status": "error", "error": str(e)}


def parse_training_result(log_file, fold_num):
    """
    解析训练日志获取结果
    这个函数会尝试从日志中提取准确率信息
    """
    result = {
        "fold": fold_num + 1,
        "status": "completed",
        "best_acc": None,
        "final_acc": None,
        "best_epoch": None
    }

    try:
        with open(log_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')

        # 尝试多种模式来解析准确率
        for line in lines:
            line = line.strip().lower()

            # 模式1: 寻找包含 "best" 和 "acc" 的行
            if "best" in line and ("acc" in line or "accuracy" in line):
                numbers = []
                words = line.split()
                for word in words:
                    try:
                        # 尝试提取数字（可能是百分比或小数）
                        if '%' in word:
                            num = float(word.replace('%', '')) / 100.0
                        else:
                            num = float(word)
                        if 0 <= num <= 1:  # 准确率应该在0-1之间
                            numbers.append(num)
                    except ValueError:
                        continue

                if numbers:
                    result["best_acc"] = max(numbers)  # 取最大值作为最佳准确率

            # 模式2: 寻找包含 "final" 和 "acc" 的行
            if "final" in line and ("acc" in line or "accuracy" in line):
                numbers = []
                words = line.split()
                for word in words:
                    try:
                        if '%' in word:
                            num = float(word.replace('%', '')) / 100.0
                        else:
                            num = float(word)
                        if 0 <= num <= 1:
                            numbers.append(num)
                    except ValueError:
                        continue

                if numbers:
                    result["final_acc"] = max(numbers)

            # 模式3: 寻找包含 "epoch" 的行
            if "epoch" in line and "best" in line:
                words = line.split()
                for i, word in enumerate(words):
                    try:
                        if word.isdigit():
                            result["best_epoch"] = int(word)
                            break
                    except ValueError:
                        continue

        # 如果没有找到best_acc，尝试从最后几行寻找任何准确率信息
        if result["best_acc"] is None:
            for line in lines[-20:]:  # 检查最后20行
                line = line.strip().lower()
                if "acc" in line or "accuracy" in line:
                    words = line.split()
                    for word in words:
                        try:
                            if '%' in word:
                                num = float(word.replace('%', '')) / 100.0
                            else:
                                num = float(word)
                            if 0 <= num <= 1:
                                result["best_acc"] = num
                                break
                        except ValueError:
                            continue
                    if result["best_acc"] is not None:
                        break

        print(f"  📊 第 {fold_num + 1} 折解析结果:")
        print(f"     - 最佳准确率: {result['best_acc']}")
        print(f"     - 最终准确率: {result['final_acc']}")
        print(f"     - 最佳轮次: {result['best_epoch']}")

    except Exception as e:
        print(f"  ⚠️ 解析第 {fold_num + 1} 折结果时出错: {e}")
        result["parse_error"] = str(e)

    return result


def summarize_results(results, save_path):
    """
    汇总交叉验证结果
    """
    print(f"\n📊 交叉验证结果汇总:")
    print("=" * 60)

    successful_folds = [r for r in results if r.get("status") == "completed" and r.get("best_acc") is not None]

    if len(successful_folds) == 0:
        print("❌ 没有成功完成的折")
        return

    # 计算统计指标
    best_accs = [r["best_acc"] for r in successful_folds]
    final_accs = [r["final_acc"] for r in successful_folds if r.get("final_acc") is not None]

    summary = {
        "total_folds": len(results),
        "successful_folds": len(successful_folds),
        "failed_folds": len(results) - len(successful_folds),
        "best_acc_mean": np.mean(best_accs),
        "best_acc_std": np.std(best_accs),
        "best_acc_min": np.min(best_accs),
        "best_acc_max": np.max(best_accs),
        "results": results
    }

    if final_accs:
        summary.update({
            "final_acc_mean": np.mean(final_accs),
            "final_acc_std": np.std(final_accs),
        })

    # 打印结果
    print(f"成功完成的折数: {summary['successful_folds']}/{summary['total_folds']}")
    print(f"\n最佳准确率统计:")
    print(f"  平均值: {summary['best_acc_mean']:.4f} ± {summary['best_acc_std']:.4f}")
    print(f"  范围: [{summary['best_acc_min']:.4f}, {summary['best_acc_max']:.4f}]")

    if final_accs:
        print(f"\n最终准确率统计:")
        print(f"  平均值: {summary['final_acc_mean']:.4f} ± {summary['final_acc_std']:.4f}")

    print(f"\n各折详细结果:")
    for r in results:
        status = r.get("status", "unknown")
        if status == "completed":
            best_acc = r.get('best_acc', 'N/A')
            if best_acc != 'N/A':
                print(f"  折 {r['fold']}: 最佳准确率 = {best_acc:.4f}")
            else:
                print(f"  折 {r['fold']}: 最佳准确率 = N/A")
        else:
            print(f"  折 {r['fold']}: {status}")

    # 保存结果
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n📁 详细结果已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="CAMELYON16 InstanT算法交叉验证")
    parser.add_argument("--config", required=True, help="基础配置文件路径")
    parser.add_argument("--data_dir", required=True, help="数据目录路径")
    parser.add_argument("--output_dir", default="./cv_results", help="结果输出目录")
    parser.add_argument("--num_folds", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--num_slides", type=int, default=10, help="每折选择的阳性slide数")
    parser.add_argument("--resume_fold", type=int, default=None, help="从指定折恢复（用于中断后继续）")

    args = parser.parse_args()

    # 检查参数
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        return

    if not os.path.exists(args.data_dir):
        print(f"❌ 数据目录不存在: {args.data_dir}")
        return

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"cv_{args.num_folds}fold_{timestamp}")
    config_dir = os.path.join(output_dir, "configs")
    log_dir = os.path.join(output_dir, "logs")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"🎯 CAMELYON16 InstanT算法 {args.num_folds}折交叉验证")
    print(f"  基础配置: {args.config}")
    print(f"  数据目录: {args.data_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  每折阳性slide数: {args.num_slides}")

    # 创建交叉验证slide划分
    cv_splits = create_cross_validation_splits(
        args.data_dir,
        num_folds=args.num_folds,
        num_slides_per_fold=args.num_slides
    )

    # 保存划分信息
    if cv_splits:
        splits_file = os.path.join(output_dir, "cv_splits.json")
        with open(splits_file, 'w') as f:
            json.dump(cv_splits, f, indent=2)
        print(f"📁 交叉验证划分已保存到: {splits_file}")

    # 为每一折创建配置文件
    print(f"\n📝 创建配置文件...")
    config_paths = []
    for fold in range(args.num_folds):
        cv_config_path = create_cv_config(
            args.config, fold, cv_splits, config_dir
        )
        config_paths.append(cv_config_path)

    # 运行交叉验证
    print(f"\n🚀 开始运行 {args.num_folds}折交叉验证...")
    results = []

    start_fold = args.resume_fold if args.resume_fold is not None else 0

    for fold in range(start_fold, args.num_folds):
        print(f"\n{'=' * 20} 第 {fold + 1}/{args.num_folds} 折 {'=' * 20}")

        try:
            result = run_single_fold(config_paths[fold], fold, log_dir)
            results.append(result)

            # 保存中间结果
            intermediate_results_file = os.path.join(output_dir, f"intermediate_results_fold_{fold + 1}.json")
            with open(intermediate_results_file, 'w') as f:
                json.dump({"completed_folds": fold + 1, "results": results}, f, indent=2)

        except KeyboardInterrupt:
            print(f"\n⚠️ 用户中断，已完成 {fold} 折")
            break
        except Exception as e:
            print(f"❌ 第 {fold + 1} 折运行异常: {e}")
            results.append({"fold": fold + 1, "status": "error", "error": str(e)})

    # 汇总结果
    if results:
        summary_file = os.path.join(output_dir, "cv_summary.json")
        summarize_results(results, summary_file)

    print(f"\n✅ 交叉验证完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main()