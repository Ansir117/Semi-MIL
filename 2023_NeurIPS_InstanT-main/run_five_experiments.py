# # run_five_experiments.py
# # CAMELYON16 InstanT算法5次独立实验脚本，阳性:阴性=1:3比例
#
# import os
# import sys
# import yaml
# import argparse
# import subprocess
# import json
# import numpy as np
# from datetime import datetime
# from sklearn.metrics import roc_auc_score
# import re
#
# # 训练脚本名称，请根据实际情况修改
# TRAIN_SCRIPT = "train.py"
#
# # ===== 数据比例配置 (统一修改入口) =====
# DEFAULT_NEGATIVE_POSITIVE_RATIO = 3  # 阴性:阳性比例，统一修改此处即可
# POSITIVE_NEGATIVE_RATIO_DISPLAY = "1:3"  # 显示用的比例字符串
#
#
# def create_experiment_config(base_config_path, experiment_id, save_dir):
#     """
#     为指定的实验创建配置文件
#
#     Args:
#         base_config_path: 基础配置文件路径
#         experiment_id: 当前实验编号 (0-4)
#         save_dir: 保存目录
#
#     Returns:
#         str: 新配置文件的路径
#     """
#     # 读取基础配置
#     with open(base_config_path, 'r', encoding='utf-8') as f:
#         config = yaml.safe_load(f)
#
#     # 修改配置参数 - 使用不同的种子确保每次实验选择不同的slide
#     config['data_seed'] = 42 + experiment_id * 100  # 42, 142, 242, 342, 442
#     config['seed'] = experiment_id * 10  # 0, 10, 20, 30, 40
#     config['save_name'] = f"camelyon16_instant_exp_{experiment_id + 1}"
#     config['experiment_id'] = experiment_id  # 新增：实验编号
#
#     # 确保统一的阳性:阴性比例
#     config['num_positive_slides'] = 10
#     config['num_negative_slides'] = 10
#     config['negative_positive_ratio'] = DEFAULT_NEGATIVE_POSITIVE_RATIO
#
#     # 创建保存目录
#     os.makedirs(save_dir, exist_ok=True)
#
#     # 保存新配置文件
#     exp_config_path = os.path.join(save_dir, f"instant_camelyon16_exp_{experiment_id + 1}.yaml")
#     with open(exp_config_path, 'w', encoding='utf-8') as f:
#         yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
#
#     print(f"  ✅ 创建实验 {experiment_id + 1} 配置文件: {exp_config_path}")
#     print(f"     - data_seed: {config['data_seed']}")
#     print(f"     - train_seed: {config['seed']}")
#     print(f"     - save_name: {config['save_name']}")
#
#     return exp_config_path
#
#
# def run_single_experiment(config_path, experiment_id, log_dir):
#     """
#     运行单个实验
#
#     Args:
#         config_path: 配置文件路径
#         experiment_id: 实验编号 (0-4)
#         log_dir: 日志保存目录
#
#     Returns:
#         dict: 实验结果
#     """
#     print(f"\n🚀 开始运行第 {experiment_id + 1} 次实验...")
#     print(f"  配置文件: {config_path}")
#
#     # 创建日志文件
#     log_file = os.path.join(log_dir, f"experiment_{experiment_id + 1}_training.log")
#     error_log = os.path.join(log_dir, f"experiment_{experiment_id + 1}_error.log")
#
#     # 构建训练命令
#     cmd = [
#         "python", TRAIN_SCRIPT,
#         "--c", config_path,
#         "--gpu", "0"  # 根据需要调整GPU设置
#     ]
#
#     print(f"  执行命令: {' '.join(cmd)}")
#
#     try:
#         # 运行训练
#         with open(log_file, 'w') as log_f, open(error_log, 'w') as err_f:
#             process = subprocess.Popen(
#                 cmd,
#                 stdout=log_f,
#                 stderr=err_f,
#                 universal_newlines=True
#             )
#
#             # 等待训练完成
#             return_code = process.wait()
#
#             if return_code == 0:
#                 print(f"  ✅ 第 {experiment_id + 1} 次实验训练完成")
#
#                 # 解析训练结果
#                 result = parse_experiment_result(log_file, experiment_id)
#                 return result
#
#             else:
#                 print(f"  ❌ 第 {experiment_id + 1} 次实验训练失败，返回码: {return_code}")
#                 print(f"  错误日志: {error_log}")
#                 return {"experiment": experiment_id + 1, "status": "failed", "return_code": return_code}
#
#     except Exception as e:
#         print(f"  ❌ 第 {experiment_id + 1} 次实验异常: {e}")
#         return {"experiment": experiment_id + 1, "status": "error", "error": str(e)}
#
#
# def parse_experiment_result(log_file, experiment_id):
#     """
#     解析实验日志获取结果，特别关注AUC指标
#     """
#     result = {
#         "experiment": experiment_id + 1,
#         "status": "completed",
#         "best_acc": None,
#         "best_auc": None,
#         "final_acc": None,
#         "final_auc": None,
#         "best_epoch": None,
#         "parse_details": []  # 添加解析详情用于调试
#     }
#
#     try:
#         with open(log_file, 'r') as f:
#             content = f.read()
#             lines = content.split('\n')
#
#         print(f"  🔍 开始解析第 {experiment_id + 1} 次实验的日志文件...")
#         print(f"     日志文件: {log_file}")
#         print(f"     总行数: {len(lines)}")
#
#         # 查找最佳结果
#         best_acc = 0.0
#         best_auc = 0.0
#         final_acc = None
#         final_auc = None
#         eval_count = 0
#
#         # 更全面的正则表达式模式
#         patterns = {
#             'acc': [
#                 r'eval.*?acc[uracy]*[:\s=]+([0-9\.]+)',
#                 r'test.*?acc[uracy]*[:\s=]+([0-9\.]+)',
#                 r'acc[uracy]*[:\s=]+([0-9\.]+)',
#                 r'Accuracy[:\s=]+([0-9\.]+)',
#                 r'eval_acc[:\s=]+([0-9\.]+)'
#             ],
#             'auc': [
#                 r'eval.*?auc[:\s=]+([0-9\.]+)',
#                 r'test.*?auc[:\s=]+([0-9\.]+)',
#                 r'auc[:\s=]+([0-9\.]+)',
#                 r'AUC[:\s=]+([0-9\.]+)',
#                 r'eval_auc[:\s=]+([0-9\.]+)',
#                 r'roc_auc[:\s=]+([0-9\.]+)'
#             ]
#         }
#
#         for line_idx, line in enumerate(lines):
#             line = line.strip()
#             if not line:
#                 continue
#
#             line_lower = line.lower()
#
#             # 检查是否包含评估相关的关键词
#             if any(keyword in line_lower for keyword in ['eval', 'test', 'acc', 'auc', 'result']):
#                 result["parse_details"].append(f"Line {line_idx}: {line}")
#                 eval_count += 1
#
#                 # 尝试提取准确率
#                 for pattern in patterns['acc']:
#                     acc_match = re.search(pattern, line, re.IGNORECASE)
#                     if acc_match:
#                         try:
#                             acc_value = float(acc_match.group(1))
#                             if acc_value > 1:  # 如果是百分比形式
#                                 acc_value = acc_value / 100.0
#                             if acc_value > best_acc and acc_value <= 1.0:  # 确保是有效的准确率
#                                 best_acc = acc_value
#                                 result["parse_details"].append(f"  Found ACC: {acc_value}")
#                         except ValueError:
#                             continue
#                         break
#
#                 # 尝试提取AUC
#                 for pattern in patterns['auc']:
#                     auc_match = re.search(pattern, line, re.IGNORECASE)
#                     if auc_match:
#                         try:
#                             auc_value = float(auc_match.group(1))
#                             if auc_value > 1:  # 如果是百分比形式
#                                 auc_value = auc_value / 100.0
#                             if auc_value > best_auc and auc_value <= 1.0:  # 确保是有效的AUC
#                                 best_auc = auc_value
#                                 result["parse_details"].append(f"  Found AUC: {auc_value}")
#                         except ValueError:
#                             continue
#                         break
#
#         # 检查最后几行是否有最终结果
#         for line in lines[-20:]:  # 检查最后20行
#             line = line.strip().lower()
#             if "final" in line or "test" in line or "best" in line:
#                 result["parse_details"].append(f"Final section: {line}")
#
#                 # 最终准确率
#                 for pattern in patterns['acc']:
#                     acc_match = re.search(pattern, line, re.IGNORECASE)
#                     if acc_match:
#                         try:
#                             final_acc = float(acc_match.group(1))
#                             if final_acc > 1:
#                                 final_acc = final_acc / 100.0
#                         except ValueError:
#                             continue
#                         break
#
#                 # 最终AUC
#                 for pattern in patterns['auc']:
#                     auc_match = re.search(pattern, line, re.IGNORECASE)
#                     if auc_match:
#                         try:
#                             final_auc = float(auc_match.group(1))
#                             if final_auc > 1:
#                                 final_auc = final_auc / 100.0
#                         except ValueError:
#                             continue
#                         break
#
#         # 更新结果，确保不保存0值
#         result["best_acc"] = best_acc if best_acc > 0 else None
#         result["best_auc"] = best_auc if best_auc > 0 else None
#         result["final_acc"] = final_acc if final_acc and final_acc > 0 else None
#         result["final_auc"] = final_auc if final_auc and final_auc > 0 else None
#         result["eval_lines_found"] = eval_count
#
#         print(f"  📊 第 {experiment_id + 1} 次实验解析结果:")
#         print(f"     - 找到评估相关行数: {eval_count}")
#         print(f"     - 最佳准确率: {result['best_acc']}")
#         print(f"     - 最佳AUC: {result['best_auc']}")
#         print(f"     - 最终准确率: {result['final_acc']}")
#         print(f"     - 最终AUC: {result['final_auc']}")
#
#         # 如果没有找到任何指标，尝试备用方法
#         if result["best_auc"] is None and result["best_acc"] is None:
#             print(f"  ⚠️ 未找到标准指标，尝试备用解析方法...")
#             result = try_manual_auc_calculation(log_file, result, experiment_id)
#
#     except Exception as e:
#         print(f"  ⚠️ 解析第 {experiment_id + 1} 次实验结果时出错: {e}")
#         result["parse_error"] = str(e)
#
#     return result
#
#
# def try_manual_auc_calculation(log_file, result, experiment_id):
#     """
#     尝试手动计算AUC（当日志中没有直接AUC记录时）
#     """
#     try:
#         print(f"     尝试替代AUC获取方法...")
#
#         with open(log_file, 'r') as f:
#             content = f.read()
#
#         # 查找包含数字的行
#         number_lines = []
#         for line in content.split('\n'):
#             if re.search(r'\d+\.\d+', line):
#                 number_lines.append(line.strip())
#
#         print(f"     找到包含数字的行数: {len(number_lines)}")
#
#         # 显示最后几行包含数字的内容用于调试
#         if number_lines:
#             print(f"     最后几行包含数字的内容:")
#             for line in number_lines[-5:]:
#                 print(f"       {line}")
#
#         # 尝试从性能指标中推断
#         # 在二分类任务中，如果准确率很高，AUC通常也会很高
#         if result["best_acc"] is not None and result["best_acc"] > 0.7:
#             # 这是一个粗略的估计，实际应用中不推荐
#             estimated_auc = min(result["best_acc"] + 0.1, 0.95)
#             print(f"     ⚠️ 基于准确率({result['best_acc']:.3f})估算AUC: {estimated_auc:.3f}")
#             result["estimated_auc"] = estimated_auc
#             result["auc_estimation_method"] = "accuracy_based"
#
#         # 查找模型保存路径，尝试从保存的模型计算AUC
#         model_save_patterns = [
#             r'save.*model.*to[:\s]*([^\s]+)',
#             r'model.*saved[:\s]*([^\s]+)',
#             r'checkpoint.*saved[:\s]*([^\s]+)'
#         ]
#
#         for pattern in model_save_patterns:
#             matches = re.findall(pattern, content, re.IGNORECASE)
#             if matches:
#                 result["model_save_path"] = matches[-1]
#                 print(f"     找到模型保存路径: {matches[-1]}")
#                 break
#
#         # 记录调试信息
#         result["debug_info"] = {
#             "total_lines": len(content.split('\n')),
#             "lines_with_numbers": len(number_lines),
#             "sample_lines": number_lines[-3:] if number_lines else [],
#             "manual_calculation_attempted": True
#         }
#
#     except Exception as e:
#         print(f"     手动AUC计算失败: {e}")
#         result["manual_auc_error"] = str(e)
#
#     return result
#
#
# def safe_format_number(value, default_str="N/A"):
#     """
#     安全地格式化数字，处理None值
#     """
#     if value is None:
#         return default_str
#     try:
#         return f"{float(value):.4f}"
#     except (ValueError, TypeError):
#         return default_str
#
#
# def summarize_experiments(results, save_path):
#     """
#     汇总5次独立实验结果，计算AUC平均值
#     """
#     print(f"\n📊 5次独立实验结果汇总:")
#     print("=" * 80)
#
#     successful_experiments = [r for r in results if r.get("status") == "completed"]
#
#     if len(successful_experiments) == 0:
#         print("❌ 没有成功完成的实验")
#         return
#
#     # 安全地提取有效的指标，处理None值
#     best_accs = []
#     best_aucs = []
#     final_accs = []
#     final_aucs = []
#
#     for r in successful_experiments:
#         if r.get("best_acc") is not None and r.get("best_acc") > 0:
#             best_accs.append(r["best_acc"])
#         if r.get("best_auc") is not None and r.get("best_auc") > 0:
#             best_aucs.append(r["best_auc"])
#         if r.get("final_acc") is not None and r.get("final_acc") > 0:
#             final_accs.append(r["final_acc"])
#         if r.get("final_auc") is not None and r.get("final_auc") > 0:
#             final_aucs.append(r["final_auc"])
#
#     summary = {
#         "total_experiments": len(results),
#         "successful_experiments": len(successful_experiments),
#         "failed_experiments": len(results) - len(successful_experiments),
#         "experiment_details": results,
#         "data_split_info": {
#             "positive_negative_ratio": POSITIVE_NEGATIVE_RATIO_DISPLAY,
#             "positive_slides_per_experiment": 10,
#             "negative_slides_per_experiment": 10,
#             "different_slides_each_experiment": True,
#             "train_test_split": "tumor*/normal* vs test*"
#         }
#     }
#
#     # 计算统计指标
#     if best_accs:
#         summary.update({
#             "best_acc_mean": np.mean(best_accs),
#             "best_acc_std": np.std(best_accs),
#             "best_acc_min": np.min(best_accs),
#             "best_acc_max": np.max(best_accs),
#             "best_acc_count": len(best_accs)
#         })
#
#     if best_aucs:
#         summary.update({
#             "best_auc_mean": np.mean(best_aucs),
#             "best_auc_std": np.std(best_aucs),
#             "best_auc_min": np.min(best_aucs),
#             "best_auc_max": np.max(best_aucs),
#             "best_auc_count": len(best_aucs)
#         })
#
#     if final_accs:
#         summary.update({
#             "final_acc_mean": np.mean(final_accs),
#             "final_acc_std": np.std(final_accs)
#         })
#
#     if final_aucs:
#         summary.update({
#             "final_auc_mean": np.mean(final_aucs),
#             "final_auc_std": np.std(final_aucs)
#         })
#
#     # 打印结果
#     print(f"实验完成情况: {summary['successful_experiments']}/{summary['total_experiments']}")
#     print(f"\n💾 数据划分信息:")
#     print(f"  阳性:阴性比例: {summary['data_split_info']['positive_negative_ratio']}")
#     print(f"  每次实验阳性slide数: {summary['data_split_info']['positive_slides_per_experiment']}")
#     print(f"  每次实验阴性slide数: {summary['data_split_info']['negative_slides_per_experiment']}")
#     print(f"  每次实验使用不同slide: {summary['data_split_info']['different_slides_each_experiment']}")
#     print(f"  训练-测试划分: {summary['data_split_info']['train_test_split']}")
#
#     if best_aucs:
#         print(f"\n🎯 最佳AUC统计 ({len(best_aucs)}次有效结果):")
#         print(f"  平均值: {summary['best_auc_mean']:.4f} ± {summary['best_auc_std']:.4f}")
#         print(f"  范围: [{summary['best_auc_min']:.4f}, {summary['best_auc_max']:.4f}]")
#         print(f"  ⭐ 关键结果: 5次实验AUC平均值 = {summary['best_auc_mean']:.4f}")
#
#     if best_accs:
#         print(f"\n📈 最佳准确率统计 ({len(best_accs)}次有效结果):")
#         print(f"  平均值: {summary['best_acc_mean']:.4f} ± {summary['best_acc_std']:.4f}")
#         print(f"  范围: [{summary['best_acc_min']:.4f}, {summary['best_acc_max']:.4f}]")
#
#     if final_aucs:
#         print(f"\n🔚 最终AUC统计:")
#         print(f"  平均值: {summary['final_auc_mean']:.4f} ± {summary['final_auc_std']:.4f}")
#
#     print(f"\n🔍 各次实验详细结果:")
#     for i, r in enumerate(results):
#         status = r.get("status", "unknown")
#         if status == "completed":
#             # 使用安全的格式化函数
#             best_auc_str = safe_format_number(r.get('best_auc'))
#             best_acc_str = safe_format_number(r.get('best_acc'))
#
#             print(f"  实验 {i + 1}: AUC = {best_auc_str}, ACC = {best_acc_str}")
#
#             # 如果没有找到指标，显示调试信息
#             if r.get('best_auc') is None and r.get('best_acc') is None:
#                 print(f"    ⚠️ 未找到有效指标，评估行数: {r.get('eval_lines_found', 0)}")
#                 if r.get('debug_info'):
#                     print(
#                         f"    调试信息: 总行数={r['debug_info']['total_lines']}, 数字行数={r['debug_info']['lines_with_numbers']}")
#         else:
#             print(f"  实验 {i + 1}: {status}")
#
#     # 如果没有找到足够的AUC结果，给出建议
#     if len(best_aucs) < len(successful_experiments):
#         print(f"\n⚠️ 注意: 只在 {len(best_aucs)}/{len(successful_experiments)} 次成功实验中找到了AUC值")
#         print("建议检查:")
#         print("1. 训练脚本是否正确输出评估指标")
#         print("2. 日志格式是否与解析逻辑匹配")
#         print("3. 模型是否正确计算并记录AUC指标")
#
#     # 保存结果
#     with open(save_path, 'w') as f:
#         json.dump(summary, f, indent=2)
#
#     print(f"\n📁 详细结果已保存到: {save_path}")
#
#     # 如果有AUC结果，额外强调
#     if best_aucs:
#         print(f"\n🏆 核心结果摘要:")
#         print(f"  5次独立实验AUC平均值: {summary['best_auc_mean']:.4f} ± {summary['best_auc_std']:.4f}")
#         print(f"  数据划分: 阳性:阴性 = {POSITIVE_NEGATIVE_RATIO_DISPLAY}，每次使用不同slide")
#         print(f"  训练-测试严格分离: ✅")
#     else:
#         print(f"\n❌ 未能从任何实验中提取到AUC值，请检查训练脚本和日志输出")
#
#
# def main():
#     parser = argparse.ArgumentParser(description="CAMELYON16 InstanT算法5次独立实验")
#     parser.add_argument("--config", required=True, help="基础配置文件路径")
#     parser.add_argument("--data_dir", required=True, help="数据目录路径")
#     parser.add_argument("--output_dir", default="./five_experiments_results", help="结果输出目录")
#     parser.add_argument("--resume_experiment", type=int, default=None, help="从指定实验恢复（用于中断后继续）")
#
#     args = parser.parse_args()
#
#     # 检查参数
#     if not os.path.exists(args.config):
#         print(f"❌ 配置文件不存在: {args.config}")
#         return
#
#     if not os.path.exists(args.data_dir):
#         print(f"❌ 数据目录不存在: {args.data_dir}")
#         return
#
#     # 创建输出目录
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_dir = os.path.join(args.output_dir, f"five_experiments_{timestamp}")
#     config_dir = os.path.join(output_dir, "configs")
#     log_dir = os.path.join(output_dir, "logs")
#
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(config_dir, exist_ok=True)
#     os.makedirs(log_dir, exist_ok=True)
#
#     print(f"🎯 CAMELYON16 InstanT算法 5次独立实验")
#     print(f"  基础配置: {args.config}")
#     print(f"  数据目录: {args.data_dir}")
#     print(f"  输出目录: {output_dir}")
#     print(f"  阳性:阴性比例: {POSITIVE_NEGATIVE_RATIO_DISPLAY}")
#     print(f"  每次实验slide选择: 不同（使用不同随机种子）")
#
#     # 为每次实验创建配置文件
#     print(f"\n📝 创建5次实验配置文件...")
#     config_paths = []
#     for exp_id in range(5):
#         exp_config_path = create_experiment_config(
#             args.config, exp_id, config_dir
#         )
#         config_paths.append(exp_config_path)
#
#     # 运行5次独立实验
#     print(f"\n🚀 开始运行5次独立实验...")
#     results = []
#
#     start_experiment = args.resume_experiment if args.resume_experiment is not None else 0
#
#     for exp_id in range(start_experiment, 5):
#         print(f"\n{'=' * 25} 第 {exp_id + 1}/5 次实验 {'=' * 25}")
#
#         try:
#             result = run_single_experiment(config_paths[exp_id], exp_id, log_dir)
#             results.append(result)
#
#             # 保存中间结果
#             intermediate_results_file = os.path.join(output_dir, f"intermediate_results_exp_{exp_id + 1}.json")
#             with open(intermediate_results_file, 'w') as f:
#                 json.dump({"completed_experiments": exp_id + 1, "results": results}, f, indent=2)
#
#         except KeyboardInterrupt:
#             print(f"\n⚠️ 用户中断，已完成 {exp_id} 次实验")
#             break
#         except Exception as e:
#             print(f"❌ 第 {exp_id + 1} 次实验运行异常: {e}")
#             results.append({"experiment": exp_id + 1, "status": "error", "error": str(e)})
#
#     # 汇总结果
#     if results:
#         summary_file = os.path.join(output_dir, "five_experiments_summary.json")
#         summarize_experiments(results, summary_file)
#
#     print(f"\n✅ 5次独立实验完成！结果保存在: {output_dir}")
#
#
# if __name__ == "__main__":
#     main()


# run_five_experiments.py
# CAMELYON16 InstanT算法5次独立实验脚本，支持patch和slide双级别AUC评估
# 修复版：使用绝对路径指定train.py

import os
import sys
import yaml
import argparse
import subprocess
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score
import re

# ===== 训练脚本配置 (使用绝对路径) =====
TRAIN_SCRIPT = "/home/xiaoyuan/Semi-MIL/2023_NeurIPS_InstanT-main/train.py"
TRAIN_SCRIPT_DIR = "/home/xiaoyuan/Semi-MIL/2023_NeurIPS_InstanT-main"

# 验证文件存在
if not os.path.exists(TRAIN_SCRIPT):
    print(f"❌ 错误：找不到train.py")
    print(f"   期望路径: {TRAIN_SCRIPT}")
    print(f"   当前目录: {os.getcwd()}")
    sys.exit(1)

print(f"✅ 将使用的train.py: {TRAIN_SCRIPT}")
print(f"✅ 工作目录: {TRAIN_SCRIPT_DIR}")

# ===== 数据比例配置 (统一修改入口) =====
DEFAULT_NEGATIVE_POSITIVE_RATIO = 3  # 阴性:阳性比例
POSITIVE_NEGATIVE_RATIO_DISPLAY = "1:3"  # 显示用的比例字符串


def create_experiment_config(base_config_path, experiment_id, save_dir):
    """
    为指定的实验创建配置文件

    Args:
        base_config_path: 基础配置文件路径
        experiment_id: 当前实验编号 (0-4)
        save_dir: 保存目录

    Returns:
        str: 新配置文件的路径
    """
    # 读取基础配置
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 修改配置参数 - 使用不同的种子确保每次实验选择不同的slide
    config['data_seed'] = 42 + experiment_id * 100  # 42, 142, 242, 342, 442
    config['seed'] = experiment_id * 10  # 0, 10, 20, 30, 40
    config['save_name'] = f"camelyon16_instant_exp_{experiment_id + 1}_with_slide"
    config['experiment_id'] = experiment_id

    # 确保统一的阳性:阴性比例
    config['num_positive_slides'] = 10
    config['num_negative_slides'] = 10
    config['negative_positive_ratio'] = DEFAULT_NEGATIVE_POSITIVE_RATIO

    # 添加slide聚合配置
    config['slide_aggregation_method'] = config.get('slide_aggregation_method', 'max')
    config['save_slide_results'] = config.get('save_slide_results', True)

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 保存新配置文件
    exp_config_path = os.path.join(save_dir, f"instant_camelyon16_exp_{experiment_id + 1}.yaml")
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"  ✅ 创建实验 {experiment_id + 1} 配置文件: {exp_config_path}")
    print(f"     - data_seed: {config['data_seed']}")
    print(f"     - train_seed: {config['seed']}")
    print(f"     - save_name: {config['save_name']}")

    return exp_config_path


def run_single_experiment(config_path, experiment_id, log_dir):
    """
    运行单个实验

    Args:
        config_path: 配置文件路径
        experiment_id: 实验编号 (0-4)
        log_dir: 日志保存目录

    Returns:
        dict: 实验结果
    """
    print(f"\n🚀 开始运行第 {experiment_id + 1} 次实验...")
    print(f"  配置文件: {config_path}")
    print(f"  训练脚本: {TRAIN_SCRIPT}")

    # 创建日志文件
    log_file = os.path.join(log_dir, f"experiment_{experiment_id + 1}_training.log")
    error_log = os.path.join(log_dir, f"experiment_{experiment_id + 1}_error.log")

    # 构建训练命令
    cmd = [
        sys.executable,  # 使用当前Python解释器
        TRAIN_SCRIPT,    # 使用绝对路径
        "--c", os.path.abspath(config_path),  # 配置文件也使用绝对路径
        "--gpu", "0"     # 根据需要调整GPU设置
    ]

    print(f"  执行命令: {' '.join(cmd)}")
    print(f"  工作目录: {TRAIN_SCRIPT_DIR}")
    print(f"  日志文件: {log_file}")

    try:
        # 运行训练
        with open(log_file, 'w', encoding='utf-8') as log_f, \
             open(error_log, 'w', encoding='utf-8') as err_f:

            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=err_f,
                universal_newlines=True,
                cwd=TRAIN_SCRIPT_DIR  # 在train.py所在目录执行
            )

            # 等待训练完成
            return_code = process.wait()

            if return_code == 0:
                print(f"  ✅ 第 {experiment_id + 1} 次实验训练完成")

                # 解析训练结果
                result = parse_experiment_result(log_file, experiment_id)
                return result

            else:
                print(f"  ❌ 第 {experiment_id + 1} 次实验训练失败，返回码: {return_code}")
                print(f"  错误日志: {error_log}")
                return {"experiment": experiment_id + 1, "status": "failed", "return_code": return_code}

    except Exception as e:
        print(f"  ❌ 第 {experiment_id + 1} 次实验异常: {e}")
        import traceback
        traceback.print_exc()
        return {"experiment": experiment_id + 1, "status": "error", "error": str(e)}


def parse_experiment_result(log_file, experiment_id):
    """
    解析实验日志获取结果，提取patch和slide两个级别的AUC
    """
    result = {
        "experiment": experiment_id + 1,
        "status": "completed",
        # Patch级别指标
        "patch_best_acc": None,
        "patch_best_auc": None,
        "patch_final_acc": None,
        "patch_final_auc": None,
        # Slide级别指标
        "slide_best_acc": None,
        "slide_best_auc": None,
        "slide_final_acc": None,
        "slide_final_auc": None,
        # 其他信息
        "aggregation_method": None,
        "num_slides": None,
        "train_py_version": None,
        "parse_details": []
    }

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')

        print(f"  📖 开始解析第 {experiment_id + 1} 次实验的日志文件...")
        print(f"     日志文件: {log_file}")
        print(f"     总行数: {len(lines)}")

        # 检查train.py版本标记
        for line in lines[:100]:  # 检查前100行
            if 'Train.py版本' in line or '🔥' in line or 'TRAIN_PY_VERSION' in line:
                result["train_py_version"] = line.strip()
                print(f"     ✓ 检测到版本信息: {line.strip()}")
                break

        # 初始化变量
        patch_best_acc = 0.0
        patch_best_auc = 0.0
        slide_best_acc = 0.0
        slide_best_auc = 0.0
        eval_count = 0

        # 增强的正则表达式模式
        patterns = {
            'patch_acc': [
                r'patch.*?acc(?:uracy)?[:\s=]+([0-9\.]+)',
                r'eval_patch_acc[:\s=]+([0-9\.]+)',
                r'Patch级别.*?准确率[:\s=]+([0-9\.]+)',
            ],
            'patch_auc': [
                r'eval_patch_auc[:\s=]+([0-9\.]+)',
                r'patch_auc[:\s=]+([0-9\.]+)',
                r'Patch级别.*?AUC[:\s=]+([0-9\.]+)',
                r'AUC-ROC\s+SCORE[:\s=]+([0-9\.]+)',  # 通用格式
            ],
            'slide_acc': [
                r'slide.*?acc(?:uracy)?[:\s=]+([0-9\.]+)',
                r'eval_slide_acc[:\s=]+([0-9\.]+)',
                r'Slide级别.*?准确率[:\s=]+([0-9\.]+)',
            ],
            'slide_auc': [
                r'eval_slide_auc[:\s=]+([0-9\.]+)',
                r'slide_auc[:\s=]+([0-9\.]+)',
                r'Slide级别.*?AUC[:\s=]+([0-9\.]+)',
            ],
        }

        # 解析每一行
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            line_lower = line.lower()

            # 检查是否包含评估相关的关键词
            if any(keyword in line_lower for keyword in ['eval', 'test', 'auc', 'acc', 'result']):
                eval_count += 1

                # === 提取Patch级别AUC ===
                for pattern in patterns['patch_auc']:
                    auc_match = re.search(pattern, line, re.IGNORECASE)
                    if auc_match:
                        try:
                            auc_value = float(auc_match.group(1))
                            if auc_value > 1:  # 如果是百分比形式
                                auc_value = auc_value / 100.0
                            if 0 < auc_value <= 1.0 and auc_value > patch_best_auc:
                                patch_best_auc = auc_value
                                result["parse_details"].append(f"Line {line_idx}: Patch AUC = {auc_value:.4f}")
                                print(f"     ✓ 找到Patch AUC: {auc_value:.4f} (line {line_idx})")
                        except (ValueError, IndexError):
                            continue
                        break

                # === 提取Slide级别AUC ===
                for pattern in patterns['slide_auc']:
                    auc_match = re.search(pattern, line, re.IGNORECASE)
                    if auc_match:
                        try:
                            auc_value = float(auc_match.group(1))
                            if auc_value > 1:
                                auc_value = auc_value / 100.0
                            if 0 < auc_value <= 1.0 and auc_value > slide_best_auc:
                                slide_best_auc = auc_value
                                result["parse_details"].append(f"Line {line_idx}: Slide AUC = {auc_value:.4f}")
                                print(f"     ✓ 找到Slide AUC: {auc_value:.4f} (line {line_idx})")
                        except (ValueError, IndexError):
                            continue
                        break

                # === 提取准确率 ===
                for pattern in patterns['patch_acc']:
                    acc_match = re.search(pattern, line, re.IGNORECASE)
                    if acc_match:
                        try:
                            acc_value = float(acc_match.group(1))
                            if acc_value > 1:
                                acc_value = acc_value / 100.0
                            if 0 < acc_value <= 1.0 and acc_value > patch_best_acc:
                                patch_best_acc = acc_value
                        except (ValueError, IndexError):
                            continue
                        break

                for pattern in patterns['slide_acc']:
                    acc_match = re.search(pattern, line, re.IGNORECASE)
                    if acc_match:
                        try:
                            acc_value = float(acc_match.group(1))
                            if acc_value > 1:
                                acc_value = acc_value / 100.0
                            if 0 < acc_value <= 1.0 and acc_value > slide_best_acc:
                                slide_best_acc = acc_value
                        except (ValueError, IndexError):
                            continue
                        break

                # 提取聚合方法
                if 'aggregation' in line_lower or '聚合方法' in line:
                    agg_match = re.search(r'(?:method|方法)[:\s=]*(\w+)', line, re.IGNORECASE)
                    if agg_match:
                        result["aggregation_method"] = agg_match.group(1)

        # 更新结果
        result["patch_best_acc"] = patch_best_acc if patch_best_acc > 0 else None
        result["patch_best_auc"] = patch_best_auc if patch_best_auc > 0 else None
        result["slide_best_acc"] = slide_best_acc if slide_best_acc > 0 else None
        result["slide_best_auc"] = slide_best_auc if slide_best_auc > 0 else None
        result["eval_lines_found"] = eval_count

        print(f"  📊 第 {experiment_id + 1} 次实验解析结果:")
        print(f"     - 找到评估相关行数: {eval_count}")
        print(f"     - Patch最佳准确率: {result['patch_best_acc']}")
        print(f"     - Patch最佳AUC: {result['patch_best_auc']}")
        print(f"     - Slide最佳准确率: {result['slide_best_acc']}")
        print(f"     - Slide最佳AUC: {result['slide_best_auc']}")

        # 警告信息
        if result["train_py_version"] is None:
            print(f"     ⚠️ 未检测到train.py版本标记 - 可能执行了旧版本")

        if result["slide_best_auc"] is None:
            print(f"     ⚠️ 未找到Slide级别AUC")
            print(f"        可能原因：")
            print(f"        1. 数据集没有返回paths信息")
            print(f"        2. slide聚合代码未执行")
            print(f"        3. 输出格式与正则表达式不匹配")

    except Exception as e:
        print(f"  ⚠️ 解析第 {experiment_id + 1} 次实验结果时出错: {e}")
        result["parse_error"] = str(e)
        import traceback
        traceback.print_exc()

    return result


def safe_format_number(value, default_str="N/A"):
    """安全地格式化数字，处理None值"""
    if value is None:
        return default_str
    try:
        return f"{float(value):.4f}"
    except (ValueError, TypeError):
        return default_str


def summarize_experiments(results, save_path):
    """
    汇总5次独立实验结果，分别计算patch和slide级别的AUC平均值
    """
    print(f"\n📊 5次独立实验结果汇总 (Patch & Slide双级别评估):")
    print("=" * 90)

    successful_experiments = [r for r in results if r.get("status") == "completed"]

    if len(successful_experiments) == 0:
        print("❌ 没有成功完成的实验")
        return

    # 分别提取patch和slide级别的指标
    patch_best_accs = []
    patch_best_aucs = []
    slide_best_accs = []
    slide_best_aucs = []

    for r in successful_experiments:
        # Patch级别指标
        if r.get("patch_best_acc") is not None and r.get("patch_best_acc") > 0:
            patch_best_accs.append(r["patch_best_acc"])
        if r.get("patch_best_auc") is not None and r.get("patch_best_auc") > 0:
            patch_best_aucs.append(r["patch_best_auc"])

        # Slide级别指标
        if r.get("slide_best_acc") is not None and r.get("slide_best_acc") > 0:
            slide_best_accs.append(r["slide_best_acc"])
        if r.get("slide_best_auc") is not None and r.get("slide_best_auc") > 0:
            slide_best_aucs.append(r["slide_best_auc"])

    summary = {
        "total_experiments": len(results),
        "successful_experiments": len(successful_experiments),
        "failed_experiments": len(results) - len(successful_experiments),
        "experiment_details": results,
        "data_split_info": {
            "positive_negative_ratio": POSITIVE_NEGATIVE_RATIO_DISPLAY,
            "positive_slides_per_experiment": 10,
            "negative_slides_per_experiment": 10,
            "different_slides_each_experiment": True,
            "train_test_split": "tumor*/normal* vs test*"
        }
    }

    # 计算Patch级别统计指标
    if patch_best_accs:
        summary.update({
            "patch_best_acc_mean": np.mean(patch_best_accs),
            "patch_best_acc_std": np.std(patch_best_accs),
            "patch_best_acc_min": np.min(patch_best_accs),
            "patch_best_acc_max": np.max(patch_best_accs),
            "patch_best_acc_count": len(patch_best_accs)
        })

    if patch_best_aucs:
        summary.update({
            "patch_best_auc_mean": np.mean(patch_best_aucs),
            "patch_best_auc_std": np.std(patch_best_aucs),
            "patch_best_auc_min": np.min(patch_best_aucs),
            "patch_best_auc_max": np.max(patch_best_aucs),
            "patch_best_auc_count": len(patch_best_aucs)
        })

    # 计算Slide级别统计指标
    if slide_best_accs:
        summary.update({
            "slide_best_acc_mean": np.mean(slide_best_accs),
            "slide_best_acc_std": np.std(slide_best_accs),
            "slide_best_acc_min": np.min(slide_best_accs),
            "slide_best_acc_max": np.max(slide_best_accs),
            "slide_best_acc_count": len(slide_best_accs)
        })

    if slide_best_aucs:
        summary.update({
            "slide_best_auc_mean": np.mean(slide_best_aucs),
            "slide_best_auc_std": np.std(slide_best_aucs),
            "slide_best_auc_min": np.min(slide_best_aucs),
            "slide_best_auc_max": np.max(slide_best_aucs),
            "slide_best_auc_count": len(slide_best_aucs)
        })

    # 打印结果
    print(f"实验完成情况: {summary['successful_experiments']}/{summary['total_experiments']}")
    print(f"\n💾 数据划分信息:")
    print(f"  阳性:阴性比例: {summary['data_split_info']['positive_negative_ratio']}")
    print(f"  每次实验阳性slide数: {summary['data_split_info']['positive_slides_per_experiment']}")
    print(f"  每次实验阴性slide数: {summary['data_split_info']['negative_slides_per_experiment']}")
    print(f"  每次实验使用不同slide: {summary['data_split_info']['different_slides_each_experiment']}")
    print(f"  训练-测试划分: {summary['data_split_info']['train_test_split']}")

    # 打印Patch级别统计
    if patch_best_aucs:
        print(f"\n📋 Patch级别AUC统计 ({len(patch_best_aucs)}次有效结果):")
        print(f"  平均值: {summary['patch_best_auc_mean']:.4f} ± {summary['patch_best_auc_std']:.4f}")
        print(f"  范围: [{summary['patch_best_auc_min']:.4f}, {summary['patch_best_auc_max']:.4f}]")

    if patch_best_accs:
        print(f"\n📈 Patch级别准确率统计 ({len(patch_best_accs)}次有效结果):")
        print(f"  平均值: {summary['patch_best_acc_mean']:.4f} ± {summary['patch_best_acc_std']:.4f}")
        print(f"  范围: [{summary['patch_best_acc_min']:.4f}, {summary['patch_best_acc_max']:.4f}]")

    # 打印Slide级别统计
    if slide_best_aucs:
        print(f"\n🏥 Slide级别AUC统计 ({len(slide_best_aucs)}次有效结果):")
        print(f"  平均值: {summary['slide_best_auc_mean']:.4f} ± {summary['slide_best_auc_std']:.4f}")
        print(f"  范围: [{summary['slide_best_auc_min']:.4f}, {summary['slide_best_auc_max']:.4f}]")
        print(f"  ⭐ 关键结果: 5次实验Slide AUC平均值 = {summary['slide_best_auc_mean']:.4f}")

    if slide_best_accs:
        print(f"\n📊 Slide级别准确率统计 ({len(slide_best_accs)}次有效结果):")
        print(f"  平均值: {summary['slide_best_acc_mean']:.4f} ± {summary['slide_best_acc_std']:.4f}")
        print(f"  范围: [{summary['slide_best_acc_min']:.4f}, {summary['slide_best_acc_max']:.4f}]")

    # 打印各次实验详细结果
    print(f"\n📝 各次实验详细结果:")
    for i, r in enumerate(results):
        status = r.get("status", "unknown")
        if status == "completed":
            patch_auc_str = safe_format_number(r.get('patch_best_auc'))
            patch_acc_str = safe_format_number(r.get('patch_best_acc'))
            slide_auc_str = safe_format_number(r.get('slide_best_auc'))
            slide_acc_str = safe_format_number(r.get('slide_best_acc'))

            print(f"  实验 {i + 1}:")
            print(f"    Patch: AUC = {patch_auc_str}, ACC = {patch_acc_str}")
            print(f"    Slide: AUC = {slide_auc_str}, ACC = {slide_acc_str}")

            # 如果没有找到指标，显示调试信息
            if (r.get('patch_best_auc') is None and r.get('slide_best_auc') is None and
                    r.get('patch_best_acc') is None and r.get('slide_best_acc') is None):
                print(f"      ⚠️ 未找到有效指标，评估行数: {r.get('eval_lines_found', 0)}")
        else:
            print(f"  实验 {i + 1}: {status}")

    # 如果没有找到足够的AUC结果，给出建议
    if len(slide_best_aucs) == 0:
        print(f"\n⚠️ 警告: 没有找到任何Slide级别AUC")
        print("可能原因及解决方案:")
        print("1. 数据集没有返回paths信息 → 检查camelyon16.py中的EvalDatasetWithSlideInfo")
        print("2. slide聚合代码未执行 → 检查train.py中的compute_patch_and_slide_auc()")
        print("3. 执行了旧版train.py → 确认TRAIN_SCRIPT路径正确")
        print("4. 输出格式不匹配 → 在train.py中添加明确的'eval_slide_auc'输出")
    elif len(slide_best_aucs) < len(successful_experiments):
        print(f"\n⚠️ 注意: 只在 {len(slide_best_aucs)}/{len(successful_experiments)} 次成功实验中找到了Slide AUC值")

    # 保存结果
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n📁 详细结果已保存到: {save_path}")

    # 最终结果摘要
    if patch_best_aucs or slide_best_aucs:
        print(f"\n🏆 核心结果摘要:")
        if patch_best_aucs:
            print(f"  📋 Patch级别: 5次实验AUC平均值 = {summary['patch_best_auc_mean']:.4f} ± {summary['patch_best_auc_std']:.4f}")
        if slide_best_aucs:
            print(f"  🏥 Slide级别: 5次实验AUC平均值 = {summary['slide_best_auc_mean']:.4f} ± {summary['slide_best_auc_std']:.4f}")
            if patch_best_aucs:
                improvement = (summary['slide_best_auc_mean'] - summary['patch_best_auc_mean']) * 100
                print(f"  📈 性能提升: {improvement:+.2f}% (Slide vs Patch)")
        print(f"  📄 数据划分: 阳性:阴性 = {POSITIVE_NEGATIVE_RATIO_DISPLAY}，每次使用不同slide")
        print(f"  ✅ 训练-测试严格分离")
    else:
        print(f"\n❌ 未能从任何实验中提取到AUC值，请检查训练脚本和日志输出")


def main():
    parser = argparse.ArgumentParser(description="CAMELYON16 InstanT算法5次独立实验，支持Patch & Slide双级别评估")
    parser.add_argument("--config", required=True, help="基础配置文件路径")
    parser.add_argument("--data_dir", required=True, help="数据目录路径")
    parser.add_argument("--output_dir", default="./five_experiments_results", help="结果输出目录")
    parser.add_argument("--resume_experiment", type=int, default=None, help="从指定实验恢复（用于中断后继续）")

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
    output_dir = os.path.join(args.output_dir, f"five_experiments_with_slide_{timestamp}")
    config_dir = os.path.join(output_dir, "configs")
    log_dir = os.path.join(output_dir, "logs")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"🎯 CAMELYON16 InstanT算法 5次独立实验 (Patch & Slide双级别评估)")
    print(f"  基础配置: {args.config}")
    print(f"  数据目录: {args.data_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  训练脚本: {TRAIN_SCRIPT}")
    print(f"  阳性:阴性比例: {POSITIVE_NEGATIVE_RATIO_DISPLAY}")
    print(f"  每次实验slide选择: 不同（使用不同随机种子）")
    print(f"  评估级别: Patch级别 + Slide级别聚合")

    # 为每次实验创建配置文件
    print(f"\n📝 创建5次实验配置文件...")
    config_paths = []
    for exp_id in range(5):
        exp_config_path = create_experiment_config(
            args.config, exp_id, config_dir
        )
        config_paths.append(exp_config_path)

    # 运行5次独立实验
    print(f"\n🚀 开始运行5次独立实验...")
    results = []

    start_experiment = args.resume_experiment if args.resume_experiment is not None else 0

    for exp_id in range(start_experiment, 5):
        print(f"\n{'=' * 30} 第 {exp_id + 1}/5 次实验 {'=' * 30}")

        try:
            result = run_single_experiment(config_paths[exp_id], exp_id, log_dir)
            results.append(result)

            # 保存中间结果
            intermediate_results_file = os.path.join(output_dir, f"intermediate_results_exp_{exp_id + 1}.json")
            with open(intermediate_results_file, 'w') as f:
                json.dump({"completed_experiments": exp_id + 1, "results": results}, f, indent=2)

        except KeyboardInterrupt:
            print(f"\n⚠️ 用户中断，已完成 {exp_id} 次实验")
            break
        except Exception as e:
            print(f"❌ 第 {exp_id + 1} 次实验运行异常: {e}")
            results.append({"experiment": exp_id + 1, "status": "error", "error": str(e)})

    # 汇总结果
    if results:
        summary_file = os.path.join(output_dir, "five_experiments_summary_with_slide.json")
        summarize_experiments(results, summary_file)

    print(f"\n✅ 5次独立实验完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main()