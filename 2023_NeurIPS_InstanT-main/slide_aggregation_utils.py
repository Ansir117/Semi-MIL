# slide_aggregation_utils.py
# CAMELYON16 Slide级别聚合工具函数

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict


def extract_slide_info_from_paths(paths):
    """
    从patch路径中提取slide信息
    Args:
        paths: patch路径列表
    Returns:
        slide_names: slide名称列表
        patch_to_slide_mapping: patch索引到slide名称的映射
    """
    patch_to_slide_mapping = {}
    slide_names = []

    for i, path in enumerate(paths):
        # 从路径中提取slide名称 (父目录名)
        slide_name = path.split('/')[-2]  # 假设路径格式为 .../slide_name/patch.jpg
        patch_to_slide_mapping[i] = slide_name
        if slide_name not in slide_names:
            slide_names.append(slide_name)

    return slide_names, patch_to_slide_mapping


def aggregate_patches_to_slides(patch_probs, patch_labels, patch_paths, aggregation_method='max'):
    """
    将patch级别的预测聚合到slide级别

    Args:
        patch_probs: patch概率预测 (N,) 或 (N, 2)
        patch_labels: patch标签 (N,)
        patch_paths: patch路径列表 (N,)
        aggregation_method: 聚合方法 ('max', 'mean', 'top_k')

    Returns:
        slide_probs: slide级别概率预测
        slide_labels: slide级别标签
        slide_names: slide名称列表
        aggregation_info: 聚合详细信息
    """

    # 如果是二维概率，取阳性类概率
    if len(patch_probs.shape) > 1 and patch_probs.shape[1] == 2:
        patch_probs = patch_probs[:, 1]

    # 提取slide信息
    slide_names, patch_to_slide_mapping = extract_slide_info_from_paths(patch_paths)

    # 按slide分组
    slide_to_patches = defaultdict(list)
    for i, slide_name in patch_to_slide_mapping.items():
        slide_to_patches[slide_name].append(i)

    slide_probs = []
    slide_labels = []
    slide_names_ordered = []
    aggregation_info = {}

    for slide_name in sorted(slide_to_patches.keys()):
        patch_indices = slide_to_patches[slide_name]

        # 获取该slide的所有patch概率和标签
        slide_patch_probs = patch_probs[patch_indices]
        slide_patch_labels = patch_labels[patch_indices]

        # slide标签：只要有一个阳性patch就是阳性slide
        slide_label = int(slide_patch_labels.max().item())

        # 聚合patch概率到slide级别
        if aggregation_method == 'max':
            slide_prob = slide_patch_probs.max().item()
        elif aggregation_method == 'mean':
            slide_prob = slide_patch_probs.mean().item()
        elif aggregation_method == 'top_k':
            # Top-3平均
            k = min(3, len(slide_patch_probs))
            slide_prob = torch.topk(slide_patch_probs, k)[0].mean().item()
        else:
            raise ValueError(f"不支持的聚合方法: {aggregation_method}")

        slide_probs.append(slide_prob)
        slide_labels.append(slide_label)
        slide_names_ordered.append(slide_name)

        # 记录聚合信息
        aggregation_info[slide_name] = {
            'num_patches': len(patch_indices),
            'slide_label': slide_label,
            'slide_prob': slide_prob,
            'patch_probs_stats': {
                'min': slide_patch_probs.min().item(),
                'max': slide_patch_probs.max().item(),
                'mean': slide_patch_probs.mean().item(),
                'std': slide_patch_probs.std().item()
            },
            'positive_patches': int(slide_patch_labels.sum().item()),
            'total_patches': len(slide_patch_labels)
        }

    return (torch.tensor(slide_probs),
            torch.tensor(slide_labels),
            slide_names_ordered,
            aggregation_info)


def compute_slide_level_metrics(slide_probs, slide_labels, slide_names=None, logger=None):
    """
    计算slide级别的评估指标
    """
    try:
        # 确保标签是有效的
        valid_mask = slide_labels >= 0
        if valid_mask.sum() == 0:
            if logger:
                logger.warning("没有有效的slide标签")
            return None

        valid_slide_probs = slide_probs[valid_mask].cpu().numpy()
        valid_slide_labels = slide_labels[valid_mask].cpu().numpy()

        # 计算准确率
        pred_labels = (valid_slide_probs > 0.5).astype(int)
        accuracy = (pred_labels == valid_slide_labels).mean()

        # 计算AUC
        auc = None
        unique_labels = np.unique(valid_slide_labels)
        if len(unique_labels) == 2 and set(unique_labels).issubset({0, 1}):
            auc = roc_auc_score(valid_slide_labels, valid_slide_probs)

        metrics = {
            'accuracy': accuracy,
            'auc': auc,
            'num_slides': len(valid_slide_labels),
            'positive_slides': int(valid_slide_labels.sum()),
            'negative_slides': int((1 - valid_slide_labels).sum())
        }

        if logger:
            logger.info(f"Slide级别指标:")
            logger.info(f"  准确率: {accuracy:.4f}")
            if auc is not None:
                logger.info(f"  AUC: {auc:.4f}")
            logger.info(f"  总slide数: {metrics['num_slides']}")
            logger.info(f"  阳性slides: {metrics['positive_slides']}")
            logger.info(f"  阴性slides: {metrics['negative_slides']}")

        return metrics

    except Exception as e:
        if logger:
            logger.error(f"计算slide级别指标时出错: {e}")
        return None


def save_slide_results(slide_probs, slide_labels, slide_names, aggregation_info, save_path):
    """
    保存slide级别的预测结果
    """
    import json
    import os

    results = {
        'slide_predictions': {},
        'summary': {
            'total_slides': len(slide_names),
            'positive_slides': int(slide_labels.sum().item()),
            'negative_slides': int((slide_labels == 0).sum().item())
        }
    }

    for i, slide_name in enumerate(slide_names):
        results['slide_predictions'][slide_name] = {
            'probability': float(slide_probs[i].item()),
            'ground_truth': int(slide_labels[i].item()),
            'prediction': int(slide_probs[i].item() > 0.5),
            'aggregation_info': aggregation_info.get(slide_name, {})
        }

    # 保存JSON结果
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Slide级别结果已保存到: {save_path}")


class SlideAggregator:
    """
    Slide级别聚合器类，用于在训练和评估过程中进行聚合
    """

    def __init__(self, aggregation_method='max', logger=None):
        self.aggregation_method = aggregation_method
        self.logger = logger
        self.reset()

    def reset(self):
        """重置累积的数据"""
        self.all_patch_probs = []
        self.all_patch_labels = []
        self.all_patch_paths = []

    def add_batch(self, patch_probs, patch_labels, patch_paths):
        """添加一个batch的数据"""
        self.all_patch_probs.append(patch_probs.cpu())
        self.all_patch_labels.append(patch_labels.cpu())

        # 确保patch_paths是列表
        if isinstance(patch_paths, (list, tuple)):
            self.all_patch_paths.extend(patch_paths)
        else:
            self.all_patch_paths.append(patch_paths)

    def compute_slide_metrics(self):
        """计算最终的slide级别指标"""
        if not self.all_patch_probs:
            if self.logger:
                self.logger.warning("没有累积的patch数据")
            return None

        # 合并所有batch的数据
        all_probs = torch.cat(self.all_patch_probs, dim=0)
        all_labels = torch.cat(self.all_patch_labels, dim=0)

        # 聚合到slide级别
        slide_probs, slide_labels, slide_names, aggregation_info = aggregate_patches_to_slides(
            all_probs, all_labels, self.all_patch_paths, self.aggregation_method
        )

        # 计算指标
        metrics = compute_slide_level_metrics(slide_probs, slide_labels, slide_names, self.logger)

        return {
            'slide_probs': slide_probs,
            'slide_labels': slide_labels,
            'slide_names': slide_names,
            'aggregation_info': aggregation_info,
            'metrics': metrics
        }