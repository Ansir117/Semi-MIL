"""
InstanT算法 + Slide级别损失的集成
放置位置: semilearn/algorithms/instant/instant_with_slide.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, DistAlignEMAHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


@ALGORITHMS.register('instant_with_slide')
class InstanTWithSlide(AlgorithmBase):
    """InstanT算法 + Slide级别损失"""

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)

        # InstanT原有初始化
        self.init(p_cutoff=args.p_cutoff, T=args.T, hard_label=args.hard_label, ema_p=args.ema_p)
        self.warm_up_it = args.warm_up_it
        self.estimation = args.estimation
        self.scale = args.scale

        # InstanT的T estimator
        if self.estimation == "instance":
            from .T_estimator import ResNet18
            self.T_estimator = ResNet18(self.num_classes * self.num_classes, scale=self.scale).cuda(self.args.gpu)
            self.T_optimizer = torch.optim.SGD(self.T_estimator.parameters(), lr=0.001, weight_decay=5e-4, momentum=0.9)
        elif self.estimation == "class":
            from .utils import scale_t
            self.vol_T = scale_t(self.gpu, self.num_classes, scale=self.scale)
            self.optimizer_vol_T = torch.optim.SGD(self.vol_T.parameters(), lr=0.01, weight_decay=0, momentum=0.9)

        # Slide级别配置
        self.use_slide_loss = getattr(args, 'use_slide_loss', True)
        self.slide_loss_weight = getattr(args, 'slide_loss_weight', 1.0)
        self.slide_warmup_epoch = getattr(args, 'slide_warmup_epoch', 10)
        self.use_consistency_loss = getattr(args, 'use_consistency_loss', False)
        self.consistency_loss_weight = getattr(args, 'consistency_loss_weight', 0.1)
        self.unlabeled_slide_loss_ratio = getattr(args, 'unlabeled_slide_loss_ratio', 0.5)

        if self.use_slide_loss:
            from .slide_mil_module import SlideMILModule, SlideInstanceConsistencyLoss

            feature_dim = self._get_feature_dim()
            self.print_fn(f"Backbone feature dimension: {feature_dim}")

            self.slide_mil = SlideMILModule(
                feature_dim=feature_dim,
                num_classes=self.num_classes,
                aggregation_method=getattr(args, 'slide_aggregation_method', 'attention'),
                hidden_dim=getattr(args, 'slide_attention_dim', 128),
                dropout=getattr(args, 'slide_dropout', 0.1),
                gated_attention=getattr(args, 'slide_gated_attention', False),
                use_instance_norm=getattr(args, 'slide_instance_norm', False)
            ).cuda(self.gpu)

            self.slide_optimizer = torch.optim.AdamW(
                self.slide_mil.parameters(),
                lr=args.lr * getattr(args, 'slide_lr_ratio', 0.1),
                weight_decay=args.weight_decay
            )

            if self.use_consistency_loss:
                self.consistency_loss_fn = SlideInstanceConsistencyLoss(
                    temperature=getattr(args, 'consistency_temperature', 0.5),
                    aggregation=getattr(args, 'consistency_aggregation', 'mean')
                ).cuda(self.gpu)

            self.print_fn("✅ Slide-level MIL module initialized")

        self.current_epoch = 0

    def _get_feature_dim(self):
        """获取backbone输出的特征维度"""
        dummy_input = torch.randn(1, 3, self.args.img_size, self.args.img_size).cuda(self.gpu)
        with torch.no_grad():
            dummy_output = self.model(dummy_input)
            if isinstance(dummy_output, dict) and 'feat' in dummy_output:
                feature_dim = dummy_output['feat'].shape[1]
            else:
                raise ValueError("Model must return a dict with 'feat' key")
        return feature_dim

    def init(self, p_cutoff, T, hard_label=True, ema_p=0.999):
        self.p_cutoff = p_cutoff
        self.T = T
        self.use_hard_label = hard_label
        self.ema_p = ema_p

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p, p_target_type='model'),
            "DistAlignHook")
        from .utils import gt_InstantThresholdingHook
        self.register_hook(gt_InstantThresholdingHook(num_classes=self.num_classes), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        """
        训练步骤 - 同时计算patch和slide损失

        注意：需要数据集返回slide_ids
        如果数据集返回格式是 (images, label_dict)，需要适配
        """
        num_lb = y_lb.shape[0]

        # 尝试从label中提取slide_ids
        if isinstance(y_lb, dict):
            slide_ids_lb = y_lb.get('slide_id', torch.zeros(num_lb, dtype=torch.long).cuda(self.gpu))
            y_lb = y_lb.get('patch_label', y_lb)
        else:
            slide_ids_lb = torch.zeros(num_lb, dtype=torch.long).cuda(self.gpu)

        with self.amp_cm():
            # Patch级别前向传播
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)

                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)

                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb)
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']

                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']

                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']

            # Patch级别损失
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            probs_x_lb = self.compute_prob(logits_x_lb.detach())
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook",
                                           probs_x_ulb=probs_x_ulb_w,
                                           probs_x_lb=probs_x_lb)

            mask = self.call_hook("masking", "MaskingHook",
                                  logits_x_lb=probs_x_lb,
                                  logits_x_ulb=probs_x_ulb_w,
                                  x_ulb_w=x_ulb_w,
                                  softmax_x_lb=False,
                                  softmax_x_ulb=False)

            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            if self.it < self.warm_up_it:
                unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask)
            else:
                if self.estimation == "instance":
                    from .utils import forward_loss
                    unsup_loss = forward_loss(logits_x_ulb_s, pseudo_label,
                                              self.T_estimator(x_ulb_w), mask=mask)
                else:
                    from .utils import class_forward_loss
                    t = self.vol_T()
                    unsup_loss = class_forward_loss(logits_x_ulb_s, pseudo_label, t, mask=mask)

            # Slide级别损失
            slide_loss = torch.tensor(0.0).cuda(self.gpu)
            consistency_loss = torch.tensor(0.0).cuda(self.gpu)

            slide_loss_enabled = (
                    self.use_slide_loss and
                    self.it >= self.warm_up_it and
                    self.current_epoch >= self.slide_warmup_epoch
            )

            if slide_loss_enabled:
                # Labeled数据的slide损失
                unique_slides_lb = torch.unique(slide_ids_lb)
                if len(unique_slides_lb) > 1:
                    slide_logits_lb, unique_slide_ids_lb = self.slide_mil(
                        feats_x_lb.detach(),
                        slide_ids_lb
                    )

                    slide_labels_gt = []
                    for sid in unique_slide_ids_lb:
                        mask_slide = (slide_ids_lb == sid)
                        slide_label_gt = y_lb[mask_slide][0]
                        slide_labels_gt.append(slide_label_gt)
                    slide_labels_gt = torch.stack(slide_labels_gt)

                    slide_loss_lb = F.cross_entropy(slide_logits_lb, slide_labels_gt, reduction='mean')
                    slide_loss += slide_loss_lb

            # 总损失
            total_loss = (
                    sup_loss +
                    self.lambda_u * unsup_loss +
                    self.slide_loss_weight * slide_loss +
                    self.consistency_loss_weight * consistency_loss
            )

        # 反向传播和优化
        feat_dict = {
            'x_lb': feats_x_lb,
            'x_ulb_w': feats_x_ulb_w,
            'x_ulb_s': feats_x_ulb_s
        }

        out_dict = self.process_out_dict(
            loss=total_loss,
            xu_loss=unsup_loss,
            slide_loss=slide_loss,
            consistency_loss=consistency_loss,
            feat=feat_dict
        )

        log_dict = self.process_log_dict(
            sup_loss=sup_loss.item(),
            unsup_loss=unsup_loss.item(),
            slide_loss=slide_loss.item(),
            consistency_loss=consistency_loss.item(),
            total_loss=total_loss.item(),
            util_ratio=mask.float().mean().item(),
            slide_loss_enabled=float(slide_loss_enabled)
        )

        self.out_dict = out_dict

        # 更新InstanT的T estimator
        if self.estimation == "instance":
            self.InstanT_update()
        else:
            self.T_update()

        # 更新Slide MIL模块
        if slide_loss_enabled and slide_loss.item() > 0:
            self.slide_optimizer.zero_grad()
            slide_loss.backward(retain_graph=False)
            self.slide_optimizer.step()

        return out_dict, log_dict

    def InstanT_update(self):
        """更新InstanT的instance-level T estimator"""
        self.T_optimizer.zero_grad()
        unsup_loss = self.out_dict['xu_loss']
        unsup_loss.backward(retain_graph=True)
        self.T_optimizer.step()

    def T_update(self):
        """更新InstanT的class-level T estimator"""
        self.optimizer_vol_T.zero_grad()
        unsup_loss = self.out_dict['xu_loss']
        unsup_loss.backward(retain_graph=True)
        self.optimizer_vol_T.step()

    def train(self):
        """重写train方法以追踪epoch"""
        self.current_epoch = self.it // (self.num_train_iter // self.epoch)
        super().train()

    @staticmethod
    def get_argument():
        """返回算法特定的参数"""
        base_args = [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--warm_up_it', int, 2000),
            SSL_Argument('--estimation', str, 'class'),
            SSL_Argument('--scale', float, 1.0),
        ]

        slide_args = [
            SSL_Argument('--use_slide_loss', str2bool, True),
            SSL_Argument('--slide_loss_weight', float, 1.0),
            SSL_Argument('--slide_warmup_epoch', int, 10),
            SSL_Argument('--slide_aggregation_method', str, 'attention'),
            SSL_Argument('--slide_attention_dim', int, 128),
            SSL_Argument('--slide_dropout', float, 0.1),
            SSL_Argument('--slide_gated_attention', str2bool, False),
            SSL_Argument('--slide_instance_norm', str2bool, False),
            SSL_Argument('--slide_lr_ratio', float, 0.1),
            SSL_Argument('--unlabeled_slide_loss_ratio', float, 0.5),
            SSL_Argument('--use_consistency_loss', str2bool, False),
            SSL_Argument('--consistency_loss_weight', float, 0.1),
            SSL_Argument('--consistency_temperature', float, 0.5),
            SSL_Argument('--consistency_aggregation', str, 'mean'),
        ]

        return base_args + slide_args