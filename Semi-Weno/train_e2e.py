"""
WENO + SSL 端到端混合训练 (End-to-End Version)
================================================
此版本支持原始图像输入，可以与USB/InstanT框架更好地集成。

与基于特征的版本区别：
1. 使用原始图像作为输入，而非预提取特征
2. 支持直接加载USB训练的模型
3. Encoder使用完整的CNN backbone

作者：基于WENO框架改进
"""

import argparse
import warnings
import os
import time
import numpy as np
from PIL import Image

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms, models
from tensorboardX import SummaryWriter
import datetime
import random
from tqdm import tqdm
import copy


# 如果有原始WENO的工具函数，取消下面的注释
# import utliz
# import util


# ==================== AUC计算工具 ====================
def cal_auc(labels, predictions):
    """计算AUC"""
    try:
        from sklearn.metrics import roc_auc_score
        labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        predictions = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
        # 过滤掉无效标签
        valid_mask = labels >= 0
        if valid_mask.sum() < 2:
            return 0.5
        return roc_auc_score(labels[valid_mask], predictions[valid_mask])
    except:
        return 0.5


# ==================== 模型定义 ====================
class ImageEncoder(nn.Module):
    """
    图像编码器 (ResNet backbone)
    输出512维特征
    """

    def __init__(self, backbone='resnet18', pretrained=True, feature_dim=512):
        super(ImageEncoder, self).__init__()

        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            in_features = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # 移除最后的FC层
        self.backbone.fc = nn.Identity()

        # 特征投影层
        if in_features != feature_dim:
            self.projector = nn.Sequential(
                nn.Linear(in_features, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.projector = nn.Identity()

        self.feature_dim = feature_dim

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.projector(feat)
        return feat


class DSMILAttentionHead(nn.Module):
    """
    DSMIL Attention Head (Teacher)
    输出attention scores和bag prediction
    """

    def __init__(self, input_dim=512, num_classes=2):
        super(DSMILAttentionHead, self).__init__()

        # 注意力机制
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(128, 1)

        # 实例分类器（用于attention score）
        self.instance_classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        # Bag分类器
        self.bag_classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, feat):
        """
        Args:
            feat: [N, D] 实例特征
        Returns:
            instance_attn_score: [N, 2] 实例级别attention/prediction
            bag_prediction: [1, 2] bag级别预测
            bag_feat: [1, D] 聚合后的bag特征
            attention_weights: [N, 1] 注意力权重
        """
        # 计算注意力权重
        A_V = self.attention_V(feat)  # [N, 128]
        A_U = self.attention_U(feat)  # [N, 128]
        A = self.attention_weights(A_V * A_U)  # [N, 1]
        A = torch.softmax(A, dim=0)  # 归一化

        # 聚合bag特征
        bag_feat = torch.mm(A.transpose(0, 1), feat)  # [1, D]

        # Bag预测
        bag_prediction = self.bag_classifier(bag_feat)  # [1, 2]

        # 实例级别分数（用作伪标签）
        instance_attn_score = self.instance_classifier(feat)  # [N, 2]

        return instance_attn_score, bag_prediction, bag_feat, A


class StudentHead(nn.Module):
    """
    Student分类头
    用于最终的patch级别预测
    """

    def __init__(self, input_dim=512, num_classes=2):
        super(StudentHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class SSLClassifier(nn.Module):
    """
    SSL分类器 (可从USB/InstanT加载)
    """

    def __init__(self, input_dim=512, hidden_dim=256, num_classes=2, dropout=0.5):
        super(SSLClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # EMA版本
        self.ema_classifier = copy.deepcopy(self.classifier)
        for param in self.ema_classifier.parameters():
            param.requires_grad = False

    def forward(self, x, use_ema=False):
        if use_ema:
            return self.ema_classifier(x)
        return self.classifier(x)

    def update_ema(self, momentum=0.999):
        for ema_param, param in zip(self.ema_classifier.parameters(), self.classifier.parameters()):
            ema_param.data.mul_(momentum).add_(param.data, alpha=1 - momentum)

    def get_prob(self, x, use_ema=True):
        with torch.no_grad():
            logits = self.forward(x, use_ema=use_ema)
            return torch.softmax(logits, dim=1)[:, 1]


# ==================== 数据集 ====================
# 优先使用同目录下的dataset.py
try:
    from dataset import CAMELYON16_E2E as CAMELYON16_Image, create_ssl_split, SSLPatchDataset
    print("[INFO] 使用外部数据集文件: dataset.py")
except ImportError:
    # 内置简化版数据集
    class CAMELYON16_Image(torch.utils.data.Dataset):
        """CAMELYON16图像数据集（内置版本）"""

        def __init__(self, root_dir, train=True, transform=None, return_bag=False,
                     max_bag_size=100, ssl_mode=False, labeled_ratio=0.2, ssl_seed=42, patch_label_file=""):
            import glob
            self.root_dir = root_dir
            self.train = train
            self.return_bag = return_bag
            self.max_bag_size = max_bag_size
            self.ssl_mode = ssl_mode

            if transform is None:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transform

            # 加载数据
            data_dir = os.path.join(root_dir, 'training' if train else 'testing')
            all_slides = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

            self.all_paths, self.patch_labels, self.slide_labels = [], [], []
            self.slide_indices, self.slide_names = [], []

            for slide_idx, slide_name in enumerate(all_slides):
                slide_dir = os.path.join(data_dir, slide_name)
                slide_label = 1 if ('pos' in slide_name.lower() or 'tumor' in slide_name.lower() or slide_name.endswith(
                    '_1')) else 0

                for img_path in glob.glob(os.path.join(slide_dir, "*.jpg")) + glob.glob(
                        os.path.join(slide_dir, "*.png")):
                    patch_name = os.path.basename(img_path).lower()
                    if '_pos' in patch_name or patch_name.replace('.jpg', '').replace('.png', '').endswith('_1'):
                        patch_label = 1
                    elif '_neg' in patch_name or patch_name.replace('.jpg', '').replace('.png', '').endswith('_0'):
                        patch_label = 0
                    else:
                        patch_label = 0 if slide_label == 0 else -1

                    self.all_paths.append(img_path)
                    self.patch_labels.append(patch_label)
                    self.slide_labels.append(slide_label)
                    self.slide_indices.append(slide_idx)
                    self.slide_names.append(slide_name)

            self.all_paths = np.array(self.all_paths)
            self.patch_labels = np.array(self.patch_labels)
            self.slide_labels = np.array(self.slide_labels)
            self.slide_indices = np.array(self.slide_indices)
            self.slide_names = np.array(self.slide_names)
            self.num_slides = len(all_slides)
            self.num_patches = len(self.all_paths)

            # SSL标签
            if ssl_mode and train:
                np.random.seed(ssl_seed)
                self.ssl_mask = np.zeros(self.num_patches, dtype=bool)
                labeled_slides = np.random.choice(np.unique(self.slide_indices),
                                                  max(1, int(self.num_slides * labeled_ratio)), replace=False)
                for s in labeled_slides:
                    mask = (self.slide_indices == s) & (self.patch_labels >= 0)
                    self.ssl_mask[mask] = True

            print(f"[DATA] {self.num_slides} slides, {self.num_patches} patches")

        def __len__(self):
            return self.num_slides if self.return_bag else self.num_patches

        def __getitem__(self, index):
            if self.return_bag:
                idx_patches = np.where(self.slide_indices == index)[0]
                if len(idx_patches) > self.max_bag_size:
                    idx_patches = np.random.choice(idx_patches, self.max_bag_size, replace=False)
                images = torch.stack(
                    [self.transform(Image.open(self.all_paths[i]).convert('RGB')) for i in idx_patches])
                patch_has_gt = torch.tensor((self.patch_labels[idx_patches] >= 0).astype(np.int64))
                return images, [torch.tensor(self.patch_labels[idx_patches]),
                                torch.tensor(self.slide_labels[idx_patches[0]]),
                                index, self.slide_names[idx_patches[0]], patch_has_gt], index
            else:
                img = self.transform(Image.open(self.all_paths[index]).convert('RGB'))
                return img, [torch.tensor(self.patch_labels[index]), torch.tensor(self.slide_labels[index]),
                             torch.tensor(self.slide_indices[index]), self.slide_names[index],
                             torch.tensor(1 if self.patch_labels[index] >= 0 else 0)], index


# ==================== Alpha调度 ====================
def get_alpha(epoch, total_epochs, schedule='linear', warmup=0, start=0.0, end=0.5):
    if epoch < warmup:
        return start

    progress = (epoch - warmup) / (total_epochs - warmup)

    if schedule == 'constant':
        return end
    elif schedule == 'linear':
        return start + (end - start) * progress
    elif schedule == 'cosine':
        return start + (end - start) * (1 - np.cos(np.pi * progress)) / 2
    elif schedule == 'step':
        if progress < 0.33:
            return start
        elif progress < 0.66:
            return (start + end) / 2
        else:
            return end
    return end


# ==================== 训练器 ====================
class E2EHybridTrainer:
    """端到端混合训练器"""

    def __init__(self, encoder, teacher_head, student_head, ssl_classifier,
                 opt_encoder, opt_teacher, opt_student, opt_ssl,
                 train_bag_loader, train_patch_loader, train_ssl_loader, test_patch_loader,
                 writer, device, args, **kwargs):

        self.encoder = encoder
        self.teacher_head = teacher_head
        self.student_head = student_head
        self.ssl_classifier = ssl_classifier

        self.opt_encoder = opt_encoder
        self.opt_teacher = opt_teacher
        self.opt_student = opt_student
        self.opt_ssl = opt_ssl

        self.train_bag_loader = train_bag_loader
        self.train_patch_loader = train_patch_loader
        self.train_ssl_loader = train_ssl_loader  # 专门用于SSL分类器训练
        self.test_patch_loader = test_patch_loader

        self.writer = writer
        self.device = device
        self.args = args

        # SSL相关
        self.ssl_split = kwargs.get('ssl_split', None)
        self.use_ssl_dataset = kwargs.get('use_ssl_dataset', False)

        self.current_alpha = args.ssl_alpha_start

    def train(self):
        for epoch in range(self.args.epochs):
            # 更新alpha
            self.current_alpha = get_alpha(
                epoch, self.args.epochs,
                schedule=self.args.ssl_alpha_schedule,
                warmup=self.args.ssl_warmup_epochs,
                start=self.args.ssl_alpha_start,
                end=self.args.ssl_alpha_end
            )

            self.writer.add_scalar('alpha', self.current_alpha, epoch)

            if epoch % 10 == 0:
                print(f"\n[Epoch {epoch}] Alpha: {self.current_alpha:.4f}")

            # 训练Teacher（使用所有slides）
            self._train_teacher(epoch)

            # 训练SSL分类器（使用有标签+无标签patches）
            if self.args.ssl_train_mode != 'pretrained':
                self._train_ssl(epoch)

            # 训练Student（使用所有patches + 混合伪标签）
            self._train_student_hybrid(epoch)

            # 评估
            if epoch % self.args.eval_period == 0:
                self._evaluate(epoch)

    def _train_teacher(self, epoch):
        self.encoder.train()
        self.teacher_head.train()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0
        for batch_idx, (images, labels, _) in enumerate(tqdm(self.train_bag_loader, desc='Teacher')):
            # images: [1, N, C, H, W]
            images = images.squeeze(0).to(self.device)
            slide_label = labels[1].to(self.device)

            # Forward
            feat = self.encoder(images)  # [N, D]
            instance_attn, bag_pred, _, _ = self.teacher_head(feat)

            # Loss
            loss = criterion(bag_pred, slide_label.unsqueeze(0))

            # 添加max instance loss
            max_idx = torch.argmax(instance_attn[:, 1])
            loss += 0.5 * criterion(instance_attn[max_idx:max_idx + 1], slide_label.unsqueeze(0))

            self.opt_encoder.zero_grad()
            self.opt_teacher.zero_grad()
            loss.backward()
            self.opt_encoder.step()
            self.opt_teacher.step()

            total_loss += loss.item()

        self.writer.add_scalar('loss/teacher', total_loss / len(self.train_bag_loader), epoch)

    def _train_ssl(self, epoch):
        """
        训练SSL分类器
        使用专门的ssl_loader，区分有标签和无标签数据
        """
        self.encoder.eval()  # 冻结encoder
        self.teacher_head.eval()
        self.ssl_classifier.train()

        criterion = nn.CrossEntropyLoss(reduction='none')
        total_loss = 0
        num_labeled_samples = 0
        num_unlabeled_samples = 0

        # 选择数据加载器
        loader = self.train_ssl_loader if self.train_ssl_loader is not None else self.train_patch_loader

        for batch_idx, batch in enumerate(tqdm(loader, desc='SSL')):
            # 处理两种数据格式
            if self.use_ssl_dataset and isinstance(batch, dict):
                # SSLPatchDataset返回的字典格式
                is_labeled = batch['is_labeled']

                # 分开处理有标签和无标签数据
                labeled_mask = is_labeled.bool()
                unlabeled_mask = ~labeled_mask

                # 有标签数据
                if labeled_mask.sum() > 0:
                    images_lb = batch['image'][labeled_mask].to(self.device)
                    labels_lb = batch['patch_label'][labeled_mask].to(self.device)
                    if 'has_gt' in batch:
                        has_gt_lb = batch['has_gt'][labeled_mask].to(self.device).bool()
                    else:
                        has_gt_lb = torch.ones_like(labels_lb).bool()

                    with torch.no_grad():
                        feat_lb = self.encoder(images_lb)

                    ssl_logits_lb = self.ssl_classifier(feat_lb.detach())
                    valid_lb = has_gt_lb & (labels_lb >= 0)
                    if valid_lb.sum() > 0:
                        loss_labeled = criterion(ssl_logits_lb[valid_lb], labels_lb[valid_lb].long()).mean()
                        num_labeled_samples += valid_lb.sum().item()
                    else:
                        loss_labeled = torch.tensor(0.0).to(self.device)
                else:
                    loss_labeled = torch.tensor(0.0).to(self.device)

                # 无标签数据
                if unlabeled_mask.sum() > 0:
                    images_ulb_w = batch['image_w'][unlabeled_mask].to(self.device)
                    images_ulb_s = batch['image_s'][unlabeled_mask].to(self.device)
                    slide_labels = batch['slide_label'][unlabeled_mask].to(self.device)

                    with torch.no_grad():
                        feat_ulb = self.encoder(images_ulb_w)
                        attn_scores, _, _, _ = self.teacher_head(feat_ulb)
                        attn_prob = torch.softmax(attn_scores, dim=1)[:, 1]

                        # 强增强的特征
                        feat_ulb_s = self.encoder(images_ulb_s)

                    ssl_logits_ulb = self.ssl_classifier(feat_ulb_s.detach())

                    # 阴性slide中的patch -> 标签0
                    neg_mask = slide_labels == 0
                    if neg_mask.sum() > 0:
                        loss_neg = criterion(ssl_logits_ulb[neg_mask],
                                             torch.zeros(neg_mask.sum()).long().to(self.device)).mean()
                    else:
                        loss_neg = torch.tensor(0.0).to(self.device)

                    # 阳性slide中高置信度patch
                    pos_mask = slide_labels == 1
                    if pos_mask.sum() > 0:
                        conf = attn_prob[pos_mask]
                        high_conf = (conf > self.args.pseudo_conf_threshold) | (conf < 1 - self.args.pseudo_conf_threshold)
                        if high_conf.sum() > 0:
                            targets = (conf[high_conf] > 0.5).long()
                            loss_pos = criterion(ssl_logits_ulb[pos_mask][high_conf], targets).mean()
                        else:
                            loss_pos = torch.tensor(0.0).to(self.device)
                    else:
                        loss_pos = torch.tensor(0.0).to(self.device)

                    loss_unlabeled = loss_neg + loss_pos
                    num_unlabeled_samples += unlabeled_mask.sum().item()
                else:
                    loss_unlabeled = torch.tensor(0.0).to(self.device)

            else:
                # 原始格式（tuple）
                images, labels, _ = batch
                images = images.to(self.device)
                patch_labels = labels[0].to(self.device)
                slide_labels = labels[1].to(self.device)
                patch_has_gt = labels[4].to(self.device) if len(labels) > 4 else (patch_labels >= 0).long()

                with torch.no_grad():
                    feat = self.encoder(images)
                    attn_scores, _, _, _ = self.teacher_head(feat)
                    attn_prob = torch.softmax(attn_scores, dim=1)[:, 1]

                ssl_logits = self.ssl_classifier(feat.detach())

                # 有标签样本
                has_label = (patch_has_gt > 0) & (patch_labels >= 0)
                if has_label.sum() > 0:
                    loss_labeled = criterion(ssl_logits[has_label], patch_labels[has_label].long()).mean()
                    num_labeled_samples += has_label.sum().item()
                else:
                    loss_labeled = torch.tensor(0.0).to(self.device)

                # 无标签样本
                no_label = patch_labels < 0
                if no_label.sum() > 0:
                    neg_mask = (slide_labels == 0) & no_label
                    if neg_mask.sum() > 0:
                        loss_neg = criterion(ssl_logits[neg_mask],
                                             torch.zeros(neg_mask.sum()).long().to(self.device)).mean()
                    else:
                        loss_neg = torch.tensor(0.0).to(self.device)

                    pos_mask = (slide_labels == 1) & no_label
                    if pos_mask.sum() > 0:
                        conf = attn_prob[pos_mask]
                        high_conf = (conf > self.args.pseudo_conf_threshold) | (conf < 1 - self.args.pseudo_conf_threshold)
                        if high_conf.sum() > 0:
                            targets = (conf[high_conf] > 0.5).long()
                            loss_pos = criterion(ssl_logits[pos_mask][high_conf], targets).mean()
                        else:
                            loss_pos = torch.tensor(0.0).to(self.device)
                    else:
                        loss_pos = torch.tensor(0.0).to(self.device)

                    loss_unlabeled = loss_neg + loss_pos
                    num_unlabeled_samples += no_label.sum().item()
                else:
                    loss_unlabeled = torch.tensor(0.0).to(self.device)

            loss = loss_labeled + self.args.ssl_loss_weight * loss_unlabeled

            self.opt_ssl.zero_grad()
            loss.backward()
            self.opt_ssl.step()
            self.ssl_classifier.update_ema()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        self.writer.add_scalar('loss/ssl', avg_loss, epoch)
        self.writer.add_scalar('ssl/num_labeled', num_labeled_samples, epoch)
        self.writer.add_scalar('ssl/num_unlabeled', num_unlabeled_samples, epoch)

    def _train_student_hybrid(self, epoch):
        self.encoder.train()
        self.student_head.train()
        self.teacher_head.eval()
        self.ssl_classifier.eval()

        total_loss = 0
        all_preds = []
        all_labels = []
        gt_ratio_running = 0.0

        for batch_idx, (images, labels, _) in enumerate(tqdm(self.train_patch_loader, desc='Student')):
            images = images.to(self.device)
            patch_labels = labels[0].to(self.device)
            slide_labels = labels[1].to(self.device)
            patch_has_gt = labels[4].to(self.device) if len(labels) > 4 else (patch_labels >= 0).long()

            feat = self.encoder(images)

            # 计算混合伪标签
            with torch.no_grad():
                # Attention score
                attn_scores, _, _, _ = self.teacher_head(feat)
                attn_prob = torch.softmax(attn_scores, dim=1)[:, 1]
                attn_prob = (attn_prob - attn_prob.min()) / (attn_prob.max() - attn_prob.min() + 1e-8)

                # SSL预测
                ssl_prob = self.ssl_classifier.get_prob(feat, use_ema=True)

                # 混合
                hybrid_pseudo = self.current_alpha * ssl_prob + (1 - self.current_alpha) * attn_prob

                # 负slide强制为0
                hybrid_pseudo[slide_labels == 0] = 0

            # Student预测
            student_logits = self.student_head(feat)
            student_prob = torch.softmax(student_logits, dim=1)

            # InstanT风格：真实patch标签 + 高置信伪标签
            target = hybrid_pseudo.clone()
            gt_mask = (patch_has_gt > 0) & (patch_labels >= 0)
            target[gt_mask] = patch_labels[gt_mask].float()

            pseudo_conf = torch.abs(hybrid_pseudo - 0.5) * 2.0
            pseudo_mask = (pseudo_conf >= self.args.pseudo_conf_threshold).float()
            sample_weight = (1.0 - self.args.student_supervised_weight) * pseudo_mask
            sample_weight[gt_mask] = self.args.student_supervised_weight
            if gt_mask.sum() == 0:
                sample_weight = pseudo_mask

            loss_per_sample = -1.0 * ((1-target) * torch.log(student_prob[:, 0] + 1e-5) +
                                      target * torch.log(student_prob[:, 1] + 1e-5))
            loss = (loss_per_sample * sample_weight).sum() / (sample_weight.sum() + 1e-6)

            self.opt_encoder.zero_grad()
            self.opt_student.zero_grad()
            loss.backward()
            self.opt_encoder.step()
            self.opt_student.step()

            total_loss += loss.item()

            # 收集预测用于计算AUC
            valid_mask = gt_mask
            if valid_mask.sum() > 0:
                all_preds.append(student_prob[valid_mask, 1].detach().cpu())
                all_labels.append(patch_labels[valid_mask].cpu())
            gt_ratio_running += gt_mask.float().mean().item()

        # 计算AUC
        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            auc = cal_auc(all_labels, all_preds)
            self.writer.add_scalar('auc/train_student', auc, epoch)

        self.writer.add_scalar('loss/student', total_loss / len(self.train_patch_loader), epoch)
        self.writer.add_scalar('train/gt_patch_ratio', gt_ratio_running / max(len(self.train_patch_loader), 1), epoch)

    def _evaluate(self, epoch):
        self.encoder.eval()
        self.student_head.eval()
        self.teacher_head.eval()
        self.ssl_classifier.eval()

        all_student_preds = []
        all_ssl_preds = []
        all_teacher_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels, _ in tqdm(self.test_patch_loader, desc='Eval'):
                images = images.to(self.device)
                patch_labels = labels[0]
                patch_has_gt = labels[4] if len(labels) > 4 else (patch_labels >= 0).long()

                feat = self.encoder(images)

                # Student
                student_prob = torch.softmax(self.student_head(feat), dim=1)[:, 1]

                # SSL
                ssl_prob = self.ssl_classifier.get_prob(feat, use_ema=True)

                # Teacher
                attn_scores, _, _, _ = self.teacher_head(feat)
                teacher_prob = torch.softmax(attn_scores, dim=1)[:, 1]

                valid = (patch_has_gt > 0) & (patch_labels >= 0)
                if valid.sum() > 0:
                    all_student_preds.append(student_prob[valid].cpu())
                    all_ssl_preds.append(ssl_prob[valid].cpu())
                    all_teacher_preds.append(teacher_prob[valid].cpu())
                    all_labels.append(patch_labels[valid])

        if len(all_labels) > 0:
            all_labels = torch.cat(all_labels)

            student_auc = cal_auc(all_labels, torch.cat(all_student_preds))
            ssl_auc = cal_auc(all_labels, torch.cat(all_ssl_preds))
            teacher_auc = cal_auc(all_labels, torch.cat(all_teacher_preds))

            self.writer.add_scalar('auc/test_student', student_auc, epoch)
            self.writer.add_scalar('auc/test_ssl', ssl_auc, epoch)
            self.writer.add_scalar('auc/test_teacher', teacher_auc, epoch)

            print(f"[Eval] Student AUC: {student_auc:.4f}, SSL AUC: {ssl_auc:.4f}, Teacher AUC: {teacher_auc:.4f}")


def get_args():
    parser = argparse.ArgumentParser()

    # 数据
    parser.add_argument('--data_dir', type=str, default='/path/to/CAMELYON16')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_bag_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)

    # 模型
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--pretrained', action='store_true', default=True)

    # 训练
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval_period', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    # SSL数据划分
    parser.add_argument('--labeled_slide_ratio', type=float, default=0.2,
                        help='有标签slide的比例')
    parser.add_argument('--labeled_patch_ratio', type=float, default=0.5,
                        help='每个有标签slide中采样的patch比例')

    # SSL训练
    parser.add_argument('--ssl_alpha_schedule', type=str, default='linear')
    parser.add_argument('--ssl_alpha_start', type=float, default=0.0)
    parser.add_argument('--ssl_alpha_end', type=float, default=0.5)
    parser.add_argument('--ssl_warmup_epochs', type=int, default=20)
    parser.add_argument('--ssl_train_mode', type=str, default='joint')
    parser.add_argument('--ssl_loss_weight', type=float, default=0.5)
    parser.add_argument('--ssl_pretrained_path', type=str, default=None)
    parser.add_argument('--pseudo_conf_threshold', type=float, default=0.8,
                        help='置信度阈值，超过该阈值的伪标签才会参与训练')
    parser.add_argument('--student_supervised_weight', type=float, default=0.7,
                        help='Student损失中真实patch标签样本的权重')
    parser.add_argument('--patch_label_file', type=str, default='',
                        help='真实patch标签文件(csv/tsv): patch_path,label')

    # 输出
    parser.add_argument('--exp_name', type=str, default='weno_ssl_e2e')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')

    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 实验名称
    exp_name = f"{args.exp_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f'./runs_e2e/{exp_name}')

    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ===================== 数据集创建 =====================
    print("\n" + "=" * 60)
    print("创建数据集和SSL划分")
    print("=" * 60)

    # 1. 基础数据集（用于Teacher的bag训练）
    train_bag_ds = CAMELYON16_Image(args.data_dir, train=True, transform=train_transform,
                                    return_bag=True, max_bag_size=args.max_bag_size,
                                    patch_label_file=args.patch_label_file)

    # 2. Patch数据集（用于Student和评估）
    train_patch_ds = CAMELYON16_Image(args.data_dir, train=True, transform=train_transform,
                                      return_bag=False, patch_label_file=args.patch_label_file)
    test_patch_ds = CAMELYON16_Image(args.data_dir, train=False, transform=test_transform,
                                     return_bag=False, patch_label_file=args.patch_label_file)

    # 3. 创建SSL划分（区分有标签/无标签patches）
    if 'create_ssl_split' in globals() and 'SSLPatchDataset' in globals():
        ssl_split = create_ssl_split(
            train_patch_ds,
            labeled_slide_ratio=args.labeled_slide_ratio,
            labeled_patch_ratio=args.labeled_patch_ratio,
            seed=args.seed
        )

        # 创建SSL专用数据集
        train_ssl_ds = SSLPatchDataset(
            train_patch_ds,
            ssl_split['labeled_indices'],
            ssl_split['unlabeled_indices'],
            transform_weak=train_transform,
            transform_strong=train_transform  # 可以使用更强的增强
        )
        use_ssl_dataset = True
        print(f"\n[SSL划分完成]")
        print(f"  有标签patches: {ssl_split['num_labeled']}")
        print(f"  无标签patches: {ssl_split['num_unlabeled']}")
        print(f"  有标签slides: {ssl_split['labeled_slides']}")
    else:
        print("[警告] 无法导入SSL划分函数，使用简化版本")
        train_ssl_ds = train_patch_ds
        ssl_split = None
        use_ssl_dataset = False

    # ===================== 数据加载器 =====================
    # Teacher: 使用所有slides (bag级别)
    train_bag_loader = torch.utils.data.DataLoader(
        train_bag_ds, batch_size=1, shuffle=True, num_workers=args.num_workers
    )

    # SSL分类器: 使用有标签+无标签patches (区分对待)
    train_ssl_loader = torch.utils.data.DataLoader(
        train_ssl_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    # Student: 使用所有patches
    train_patch_loader = torch.utils.data.DataLoader(
        train_patch_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    # 测试
    test_patch_loader = torch.utils.data.DataLoader(
        test_patch_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    print(f"\n[数据加载器]")
    print(f"  Teacher (bag): {len(train_bag_loader)} batches")
    print(f"  SSL分类器: {len(train_ssl_loader)} batches")
    print(f"  Student (patch): {len(train_patch_loader)} batches")
    print(f"  测试: {len(test_patch_loader)} batches")

    # ===================== 模型 =====================
    encoder = ImageEncoder(args.backbone, args.pretrained, args.feature_dim).to(device)
    teacher_head = DSMILAttentionHead(args.feature_dim).to(device)
    student_head = StudentHead(args.feature_dim).to(device)
    ssl_classifier = SSLClassifier(args.feature_dim).to(device)

    # 加载预训练SSL
    if args.ssl_pretrained_path and os.path.exists(args.ssl_pretrained_path):
        ckpt = torch.load(args.ssl_pretrained_path, map_location=device)
        if 'state_dict' in ckpt:
            ssl_classifier.load_state_dict(ckpt['state_dict'])
        else:
            ssl_classifier.load_state_dict(ckpt)
        print(f"Loaded SSL classifier from {args.ssl_pretrained_path}")

    # ===================== 优化器 =====================
    opt_encoder = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_teacher = torch.optim.AdamW(teacher_head.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_student = torch.optim.AdamW(student_head.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_ssl = torch.optim.AdamW(ssl_classifier.parameters(), lr=args.lr, weight_decay=1e-4)

    # ===================== 训练 =====================
    trainer = E2EHybridTrainer(
        encoder, teacher_head, student_head, ssl_classifier,
        opt_encoder, opt_teacher, opt_student, opt_ssl,
        train_bag_loader, train_patch_loader, train_ssl_loader, test_patch_loader,
        writer, device, args,
        ssl_split=ssl_split,
        use_ssl_dataset=use_ssl_dataset
    )

    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    trainer.train()

    # ===================== 保存 =====================
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save({
        'encoder': encoder.state_dict(),
        'teacher_head': teacher_head.state_dict(),
        'student_head': student_head.state_dict(),
        'ssl_classifier': ssl_classifier.state_dict(),
        'ssl_split': ssl_split,
    }, os.path.join(args.save_dir, f'{exp_name}.pth'))

    print(f"\nTraining completed! Model saved to {args.save_dir}/{exp_name}.pth")


if __name__ == '__main__':
    main()