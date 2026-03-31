"""
Slide-level Multiple Instance Learning (MIL) Module
包含Attention-based聚合和Slide分类头
放置位置: semilearn/algorithms/instant/slide_mil_module.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMIL(nn.Module):
    """Attention-based Multiple Instance Learning"""

    def __init__(self, feature_dim, hidden_dim=128, dropout=0.1, gated=False):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.gated = gated

        if gated:
            self.attention_V = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh()
            )
            self.attention_U = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Sigmoid()
            )
            self.attention_w = nn.Linear(hidden_dim, 1)
        else:
            self.attention_net = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, instance_features, return_attention=False):
        if self.gated:
            A_V = self.attention_V(instance_features)
            A_U = self.attention_U(instance_features)
            A = self.attention_w(A_V * A_U)
        else:
            A = self.attention_net(instance_features)

        attention_weights = F.softmax(A, dim=0)
        bag_feature = torch.sum(instance_features * attention_weights, dim=0)

        if return_attention:
            return bag_feature, attention_weights.squeeze()
        return bag_feature


class MaxPoolingMIL(nn.Module):
    """Max Pooling based MIL"""

    def forward(self, instance_features, return_attention=False):
        bag_feature, max_idx = instance_features.max(dim=0)

        if return_attention:
            attention_weights = torch.zeros(instance_features.size(0), device=instance_features.device)
            attention_weights[max_idx[0]] = 1.0
            return bag_feature, attention_weights
        return bag_feature


class MeanPoolingMIL(nn.Module):
    """Mean Pooling based MIL"""

    def forward(self, instance_features, return_attention=False):
        bag_feature = instance_features.mean(dim=0)

        if return_attention:
            attention_weights = torch.ones(instance_features.size(0), device=instance_features.device)
            attention_weights = attention_weights / attention_weights.sum()
            return bag_feature, attention_weights
        return bag_feature


class SlideMILModule(nn.Module):
    """完整的Slide MIL模块：Aggregation + Slide Classifier"""

    def __init__(
            self,
            feature_dim,
            num_classes,
            aggregation_method='attention',
            hidden_dim=128,
            dropout=0.1,
            gated_attention=False,
            use_instance_norm=False
    ):
        super().__init__()

        self.aggregation_method = aggregation_method
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.use_instance_norm = use_instance_norm

        if use_instance_norm:
            self.instance_norm = nn.LayerNorm(feature_dim)

        if aggregation_method == 'attention':
            self.aggregator = AttentionMIL(feature_dim, hidden_dim, dropout, gated=False)
        elif aggregation_method == 'attention_gated':
            self.aggregator = AttentionMIL(feature_dim, hidden_dim, dropout, gated=True)
        elif aggregation_method == 'max':
            self.aggregator = MaxPoolingMIL()
        elif aggregation_method == 'mean':
            self.aggregator = MeanPoolingMIL()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        self.slide_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.slide_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, instance_features, slide_ids, return_attention=False, return_bag_features=False):
        """
        将多个instances聚合成slides并进行分类

        Args:
            instance_features: [B, feature_dim]
            slide_ids: [B]
            return_attention: 是否返回attention权重
            return_bag_features: 是否返回bag特征

        Returns:
            slide_logits: [num_slides, num_classes]
            slide_ids_unique: [num_slides]
            extras: Dict (可选)
        """
        unique_slide_ids = torch.unique(slide_ids)
        slide_features = []
        attention_dict = {} if return_attention else None

        if self.use_instance_norm:
            instance_features = self.instance_norm(instance_features)

        for slide_id in unique_slide_ids:
            mask = (slide_ids == slide_id)
            slide_instances = instance_features[mask]

            if return_attention:
                slide_feat, attn_weights = self.aggregator(slide_instances, return_attention=True)
                attention_dict[slide_id.item()] = attn_weights.cpu().detach()
            else:
                slide_feat = self.aggregator(slide_instances, return_attention=False)

            slide_features.append(slide_feat)

        slide_features = torch.stack(slide_features)
        slide_logits = self.slide_classifier(slide_features)

        if return_attention or return_bag_features:
            extras = {}
            if return_attention:
                extras['attention_dict'] = attention_dict
            if return_bag_features:
                extras['bag_features'] = slide_features
            return slide_logits, unique_slide_ids, extras

        return slide_logits, unique_slide_ids


class SlideInstanceConsistencyLoss(nn.Module):
    """Slide-Instance一致性损失"""

    def __init__(self, temperature=0.5, aggregation='mean'):
        super().__init__()
        self.temperature = temperature
        self.aggregation = aggregation

    def forward(self, instance_probs, slide_probs, slide_ids):
        unique_slide_ids = torch.unique(slide_ids)
        losses = []

        for idx, slide_id in enumerate(unique_slide_ids):
            mask = (slide_ids == slide_id)
            slide_instance_probs = instance_probs[mask]

            if self.aggregation == 'mean':
                aggregated_probs = slide_instance_probs.mean(dim=0)
            elif self.aggregation == 'max':
                aggregated_probs = slide_instance_probs.max(dim=0)[0]
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")

            kl_loss = F.kl_div(
                F.log_softmax(slide_probs[idx] / self.temperature, dim=0),
                F.softmax(aggregated_probs / self.temperature, dim=0),
                reduction='batchmean'
            )
            losses.append(kl_loss)

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=instance_probs.device)