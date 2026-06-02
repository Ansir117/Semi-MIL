import argparse
import copy
import datetime
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataset_feat import CAMELYON16FeatureDataset, create_ssl_split_feature, SSLPatchFeatureDataset


def cal_auc(labels, predictions):
    try:
        from sklearn.metrics import roc_auc_score
        labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        predictions = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
        valid = labels >= 0
        if valid.sum() < 2:
            return 0.5
        return roc_auc_score(labels[valid], predictions[valid])
    except Exception:
        return 0.5


class FeatureProjector(nn.Module):
    def __init__(self, input_dim=2048, feature_dim=512):
        super().__init__()
        if input_dim == feature_dim:
            self.projector = nn.Identity()
        else:
            self.projector = nn.Sequential(
                nn.Linear(input_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.projector(x)


class DSMILAttentionHead(nn.Module):
    def __init__(self, input_dim=512, num_classes=2):
        super().__init__()
        self.attention_v = nn.Sequential(nn.Linear(input_dim, 128), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(input_dim, 128), nn.Sigmoid())
        self.attention_w = nn.Linear(128, 1)
        self.instance_classifier = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes)
        )
        self.bag_classifier = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes)
        )

    def forward(self, feat):
        av = self.attention_v(feat)
        au = self.attention_u(feat)
        a = self.attention_w(av * au)
        a = torch.softmax(a, dim=0)
        bag_feat = torch.mm(a.transpose(0, 1), feat)
        bag_pred = self.bag_classifier(bag_feat)
        instance_score = self.instance_classifier(feat)
        return instance_score, bag_pred, bag_feat, a


class StudentHead(nn.Module):
    def __init__(self, input_dim=512, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class SSLClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=2, dropout=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.ema_classifier = copy.deepcopy(self.classifier)
        for p in self.ema_classifier.parameters():
            p.requires_grad = False

    def forward(self, x, use_ema=False):
        return self.ema_classifier(x) if use_ema else self.classifier(x)

    def update_ema(self, momentum=0.999):
        for ep, p in zip(self.ema_classifier.parameters(), self.classifier.parameters()):
            ep.data.mul_(momentum).add_(p.data, alpha=1.0 - momentum)

    def get_prob(self, x, use_ema=True):
        with torch.no_grad():
            logits = self.forward(x, use_ema=use_ema)
            return torch.softmax(logits, dim=1)[:, 1]


def get_alpha(epoch, total_epochs, schedule="linear", warmup=0, start=0.0, end=0.5):
    if epoch < warmup:
        return start
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    if schedule == "constant":
        return end
    if schedule == "linear":
        return start + (end - start) * progress
    if schedule == "cosine":
        return start + (end - start) * (1 - np.cos(np.pi * progress)) / 2
    return end


class FeatureHybridTrainer:
    def __init__(
        self,
        projector,
        teacher_head,
        student_head,
        ssl_classifier,
        opt_projector,
        opt_teacher,
        opt_student,
        opt_ssl,
        train_bag_loader,
        train_patch_loader,
        train_ssl_loader,
        test_patch_loader,
        patch_eval_loader,
        writer,
        device,
        args,
    ):
        self.projector = projector
        self.teacher_head = teacher_head
        self.student_head = student_head
        self.ssl_classifier = ssl_classifier
        self.opt_projector = opt_projector
        self.opt_teacher = opt_teacher
        self.opt_student = opt_student
        self.opt_ssl = opt_ssl
        self.train_bag_loader = train_bag_loader
        self.train_patch_loader = train_patch_loader
        self.train_ssl_loader = train_ssl_loader
        self.test_patch_loader = test_patch_loader
        self.patch_eval_loader = patch_eval_loader
        self.writer = writer
        self.device = device
        self.args = args
        self.current_alpha = args.ssl_alpha_start
        self.best_slide_auc_student = -1.0
        os.makedirs(args.save_dir, exist_ok=True)
        self.best_result_path = os.path.join(args.save_dir, "best_test_metrics.json")

    def _safe_ssl_forward(self, feat_input):
        """
        BatchNorm-safe SSL forward.
        If effective batch size is 1, run SSL classifier in eval mode
        to avoid BatchNorm runtime error.
        """
        if feat_input.shape[0] > 1:
            return self.ssl_classifier(feat_input)
        prev_mode = self.ssl_classifier.training
        self.ssl_classifier.eval()
        out = self.ssl_classifier(feat_input)
        if prev_mode:
            self.ssl_classifier.train()
        return out

    def train(self):
        for epoch in range(self.args.epochs):
            self.current_alpha = get_alpha(
                epoch, self.args.epochs,
                schedule=self.args.ssl_alpha_schedule,
                warmup=self.args.ssl_warmup_epochs,
                start=self.args.ssl_alpha_start,
                end=self.args.ssl_alpha_end,
            )
            self.writer.add_scalar("alpha", self.current_alpha, epoch)
            self._train_teacher(epoch)
            self._train_ssl(epoch)
            self._train_student(epoch)
            if epoch % self.args.eval_period == 0:
                metrics = self._evaluate(epoch)
                self._update_best(epoch, metrics)

    def _update_best(self, epoch, metrics):
        slide_auc = float(metrics.get("slide_auc_student", -1.0))
        if slide_auc <= self.best_slide_auc_student:
            return
        self.best_slide_auc_student = slide_auc
        payload = {
            "best_epoch": int(epoch),
            "best_slide_auc_student": slide_auc,
            "best_patch_auc_student_holdout": float(metrics.get("patch_auc_student_holdout", -1.0)),
            "test_patch_auc_student": float(metrics.get("test_patch_auc_student", -1.0)),
            "test_patch_auc_ssl": float(metrics.get("test_patch_auc_ssl", -1.0)),
            "test_patch_auc_teacher": float(metrics.get("test_patch_auc_teacher", -1.0)),
            "num_slides_test": int(metrics.get("num_slides_test", 0)),
            "num_patch_test_with_gt": int(metrics.get("num_patch_test_with_gt", 0)),
            "num_patch_holdout_with_gt": int(metrics.get("num_patch_holdout_with_gt", 0)),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.best_result_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[Best Updated] epoch={epoch}, slide_auc_student={slide_auc:.4f}, json={self.best_result_path}")

    def _train_teacher(self, epoch):
        self.projector.train()
        self.teacher_head.train()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        for feat_bag, labels, _ in tqdm(self.train_bag_loader, desc="Teacher"):
            feat_bag = feat_bag.squeeze(0).to(self.device)
            slide_label = labels[1].to(self.device).long().view(-1)
            feat = self.projector(feat_bag)
            inst_score, bag_pred, _, _ = self.teacher_head(feat)
            loss = criterion(bag_pred, slide_label)
            max_idx = torch.argmax(inst_score[:, 1])
            loss += 0.5 * criterion(inst_score[max_idx:max_idx + 1], slide_label)
            self.opt_projector.zero_grad()
            self.opt_teacher.zero_grad()
            loss.backward()
            self.opt_projector.step()
            self.opt_teacher.step()
            total_loss += loss.item()
        self.writer.add_scalar("loss/teacher", total_loss / max(len(self.train_bag_loader), 1), epoch)

    def _train_ssl(self, epoch):
        self.projector.eval()
        self.teacher_head.eval()
        self.ssl_classifier.train()
        criterion = nn.CrossEntropyLoss(reduction="none")
        total_loss = 0.0
        for batch in tqdm(self.train_ssl_loader, desc="SSL"):
            is_labeled = batch["is_labeled"].bool()
            labeled_mask = is_labeled
            unlabeled_mask = ~is_labeled

            loss_labeled = torch.tensor(0.0, device=self.device)
            if labeled_mask.sum() > 0:
                feat_lb = batch["feat"][labeled_mask].to(self.device)
                labels_lb = batch["patch_label"][labeled_mask].to(self.device)
                has_gt_lb = batch["has_gt"][labeled_mask].to(self.device).bool()
                with torch.no_grad():
                    proj_lb = self.projector(feat_lb)
                ssl_logits_lb = self._safe_ssl_forward(proj_lb.detach())
                valid = has_gt_lb & (labels_lb >= 0)
                if valid.sum() > 0:
                    loss_labeled = criterion(ssl_logits_lb[valid], labels_lb[valid].long()).mean()

            loss_unlabeled = torch.tensor(0.0, device=self.device)
            if unlabeled_mask.sum() > 0:
                feat_w = batch["feat_w"][unlabeled_mask].to(self.device)
                feat_s = batch["feat_s"][unlabeled_mask].to(self.device)
                slide_lb = batch["slide_label"][unlabeled_mask].to(self.device)
                with torch.no_grad():
                    proj_w = self.projector(feat_w)
                    attn_score, _, _, _ = self.teacher_head(proj_w)
                    attn_prob = torch.softmax(attn_score, dim=1)[:, 1]
                    proj_s = self.projector(feat_s)
                ssl_logits_ulb = self._safe_ssl_forward(proj_s.detach())
                neg_mask = slide_lb == 0
                loss_neg = torch.tensor(0.0, device=self.device)
                if neg_mask.sum() > 0:
                    loss_neg = criterion(ssl_logits_ulb[neg_mask], torch.zeros(neg_mask.sum(), dtype=torch.long, device=self.device)).mean()
                pos_mask = slide_lb == 1
                loss_pos = torch.tensor(0.0, device=self.device)
                if pos_mask.sum() > 0:
                    conf = attn_prob[pos_mask]
                    high_conf = (conf > self.args.pseudo_conf_threshold) | (conf < 1 - self.args.pseudo_conf_threshold)
                    if high_conf.sum() > 0:
                        targets = (conf[high_conf] > 0.5).long()
                        loss_pos = criterion(ssl_logits_ulb[pos_mask][high_conf], targets).mean()
                loss_unlabeled = loss_neg + loss_pos

            loss = loss_labeled + self.args.ssl_loss_weight * loss_unlabeled
            self.opt_ssl.zero_grad()
            loss.backward()
            self.opt_ssl.step()
            self.ssl_classifier.update_ema()
            total_loss += loss.item()
        self.writer.add_scalar("loss/ssl", total_loss / max(len(self.train_ssl_loader), 1), epoch)

    def _train_student(self, epoch):
        self.projector.train()
        self.student_head.train()
        self.teacher_head.eval()
        self.ssl_classifier.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        for feat_raw, labels, _ in tqdm(self.train_patch_loader, desc="Student"):
            feat_raw = feat_raw.to(self.device)
            patch_labels = labels[0].to(self.device)
            slide_labels = labels[1].to(self.device)
            patch_has_gt = labels[4].to(self.device)

            feat = self.projector(feat_raw)
            with torch.no_grad():
                attn_score, _, _, _ = self.teacher_head(feat)
                attn_prob = torch.softmax(attn_score, dim=1)[:, 1]
                attn_prob = (attn_prob - attn_prob.min()) / (attn_prob.max() - attn_prob.min() + 1e-8)
                ssl_prob = self.ssl_classifier.get_prob(feat, use_ema=True)
                hybrid_pseudo = self.current_alpha * ssl_prob + (1 - self.current_alpha) * attn_prob
                hybrid_pseudo[slide_labels == 0] = 0

            student_logits = self.student_head(feat)
            student_prob = torch.softmax(student_logits, dim=1)

            target = hybrid_pseudo.clone()
            gt_mask = (patch_has_gt > 0) & (patch_labels >= 0)
            target[gt_mask] = patch_labels[gt_mask].float()
            pseudo_conf = torch.abs(hybrid_pseudo - 0.5) * 2.0
            pseudo_mask = (pseudo_conf >= self.args.pseudo_conf_threshold).float()
            sample_weight = (1.0 - self.args.student_supervised_weight) * pseudo_mask
            sample_weight[gt_mask] = self.args.student_supervised_weight
            if gt_mask.sum() == 0:
                sample_weight = pseudo_mask
            loss_per = -1.0 * ((1 - target) * torch.log(student_prob[:, 0] + 1e-5) + target * torch.log(student_prob[:, 1] + 1e-5))
            loss = (loss_per * sample_weight).sum() / (sample_weight.sum() + 1e-6)

            self.opt_projector.zero_grad()
            self.opt_student.zero_grad()
            loss.backward()
            self.opt_projector.step()
            self.opt_student.step()
            total_loss += loss.item()

            if gt_mask.sum() > 0:
                all_preds.append(student_prob[gt_mask, 1].detach().cpu())
                all_labels.append(patch_labels[gt_mask].cpu())

        if len(all_preds) > 0:
            auc = cal_auc(torch.cat(all_labels), torch.cat(all_preds))
            self.writer.add_scalar("auc/train_student", auc, epoch)
        self.writer.add_scalar("loss/student", total_loss / max(len(self.train_patch_loader), 1), epoch)

    def _evaluate(self, epoch):
        self.projector.eval()
        self.teacher_head.eval()
        self.student_head.eval()
        self.ssl_classifier.eval()
        all_student, all_ssl, all_teacher, all_gt = [], [], [], []
        test_slide_ids = []
        test_slide_labels = []
        test_student_prob = []
        with torch.no_grad():
            for feat_raw, labels, _ in tqdm(self.test_patch_loader, desc="Eval"):
                feat_raw = feat_raw.to(self.device)
                patch_labels = labels[0]
                slide_labels = labels[1]
                slide_ids = labels[2]
                patch_has_gt = labels[4]
                feat = self.projector(feat_raw)
                student_prob = torch.softmax(self.student_head(feat), dim=1)[:, 1]
                ssl_prob = self.ssl_classifier.get_prob(feat, use_ema=True)
                attn_score, _, _, _ = self.teacher_head(feat)
                teacher_prob = torch.softmax(attn_score, dim=1)[:, 1]

                test_slide_ids.append(slide_ids.cpu())
                test_slide_labels.append(slide_labels.cpu())
                test_student_prob.append(student_prob.cpu())
                valid = (patch_has_gt > 0) & (patch_labels >= 0)
                if valid.sum() > 0:
                    all_student.append(student_prob[valid].cpu())
                    all_ssl.append(ssl_prob[valid].cpu())
                    all_teacher.append(teacher_prob[valid].cpu())
                    all_gt.append(patch_labels[valid])
        if len(all_gt) > 0:
            gt = torch.cat(all_gt)
            test_patch_auc_student = cal_auc(gt, torch.cat(all_student))
            test_patch_auc_ssl = cal_auc(gt, torch.cat(all_ssl))
            test_patch_auc_teacher = cal_auc(gt, torch.cat(all_teacher))
            num_patch_test_with_gt = int(gt.shape[0])
        else:
            test_patch_auc_student = 0.5
            test_patch_auc_ssl = 0.5
            test_patch_auc_teacher = 0.5
            num_patch_test_with_gt = 0
        self.writer.add_scalar("auc/test_patch_student", test_patch_auc_student, epoch)
        self.writer.add_scalar("auc/test_patch_ssl", test_patch_auc_ssl, epoch)
        self.writer.add_scalar("auc/test_patch_teacher", test_patch_auc_teacher, epoch)

        slide_ids_all = torch.cat(test_slide_ids).numpy().astype(np.int64)
        slide_labels_all = torch.cat(test_slide_labels).numpy().astype(np.int64)
        student_prob_all = torch.cat(test_student_prob).numpy()
        slide_pred_dict = {}
        slide_label_dict = {}
        for sid, slb, prob in zip(slide_ids_all, slide_labels_all, student_prob_all):
            if sid not in slide_pred_dict:
                slide_pred_dict[sid] = float(prob)
                slide_label_dict[sid] = int(slb)
            else:
                slide_pred_dict[sid] = max(slide_pred_dict[sid], float(prob))
        slide_gt = np.array([slide_label_dict[sid] for sid in sorted(slide_pred_dict.keys())], dtype=np.int64)
        slide_pred = np.array([slide_pred_dict[sid] for sid in sorted(slide_pred_dict.keys())], dtype=np.float32)
        slide_auc_student = cal_auc(slide_gt, slide_pred)
        self.writer.add_scalar("auc/test_slide_student", slide_auc_student, epoch)

        patch_auc_student_holdout = 0.5
        num_patch_holdout_with_gt = 0
        if self.patch_eval_loader is not None:
            holdout_pred = []
            holdout_gt = []
            with torch.no_grad():
                for feat_raw, labels, _ in self.patch_eval_loader:
                    feat_raw = feat_raw.to(self.device)
                    patch_labels = labels[0]
                    patch_has_gt = labels[4]
                    feat = self.projector(feat_raw)
                    student_prob = torch.softmax(self.student_head(feat), dim=1)[:, 1]
                    valid = (patch_has_gt > 0) & (patch_labels >= 0)
                    if valid.sum() > 0:
                        holdout_pred.append(student_prob[valid].cpu())
                        holdout_gt.append(patch_labels[valid])
            if len(holdout_gt) > 0:
                holdout_gt = torch.cat(holdout_gt)
                holdout_pred = torch.cat(holdout_pred)
                patch_auc_student_holdout = cal_auc(holdout_gt, holdout_pred)
                num_patch_holdout_with_gt = int(holdout_gt.shape[0])
            self.writer.add_scalar("auc/patch_holdout_student", patch_auc_student_holdout, epoch)

        print(
            f"[Eval][Epoch {epoch}] "
            f"slide_auc_student={slide_auc_student:.4f}, "
            f"test_patch_auc_student={test_patch_auc_student:.4f}, "
            f"holdout_patch_auc_student={patch_auc_student_holdout:.4f}"
        )
        return {
            "slide_auc_student": slide_auc_student,
            "patch_auc_student_holdout": patch_auc_student_holdout,
            "test_patch_auc_student": test_patch_auc_student,
            "test_patch_auc_ssl": test_patch_auc_ssl,
            "test_patch_auc_teacher": test_patch_auc_teacher,
            "num_slides_test": len(slide_pred_dict),
            "num_patch_test_with_gt": num_patch_test_with_gt,
            "num_patch_holdout_with_gt": num_patch_holdout_with_gt,
        }


def get_args():
    p = argparse.ArgumentParser(description="Semi-WENO feature-based training")
    p.add_argument("--feature_root", type=str, required=True, help="Directory that contains training_features.npz/testing_features.npz")
    p.add_argument("--feature_input_dim", type=int, default=2048, help="Input feature dim (ResNet50=2048)")
    p.add_argument("--feature_dim", type=int, default=512, help="Projected feature dim used by teacher/student/ssl")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_bag_size", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval_period", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--labeled_slide_ratio", type=float, default=0.2)
    p.add_argument("--labeled_patch_ratio", type=float, default=0.5)
    p.add_argument("--ssl_alpha_schedule", type=str, default="linear")
    p.add_argument("--ssl_alpha_start", type=float, default=0.0)
    p.add_argument("--ssl_alpha_end", type=float, default=0.5)
    p.add_argument("--ssl_warmup_epochs", type=int, default=20)
    p.add_argument("--ssl_loss_weight", type=float, default=0.5)
    p.add_argument("--pseudo_conf_threshold", type=float, default=0.8)
    p.add_argument("--student_supervised_weight", type=float, default=0.7)
    p.add_argument("--strong_noise_std", type=float, default=0.05)
    p.add_argument("--patch_eval_holdout_ratio", type=float, default=0.1,
                   help="Reserve this ratio of labeled training patches for patch-level evaluation")

    p.add_argument("--exp_name", type=str, default="weno_ssl_feat")
    p.add_argument("--save_dir", type=str, default="./checkpoints_feat")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    exp_name = f"{args.exp_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f"./runs_feat/{exp_name}")

    train_bag_ds = CAMELYON16FeatureDataset(args.feature_root, train=True, return_bag=True, max_bag_size=args.max_bag_size)
    train_patch_ds = CAMELYON16FeatureDataset(args.feature_root, train=True, return_bag=False)
    test_patch_ds = CAMELYON16FeatureDataset(args.feature_root, train=False, return_bag=False)

    all_train_indices = np.arange(train_patch_ds.num_patches, dtype=np.int64)
    labeled_train_idx = np.where((train_patch_ds.patch_has_gt == 1) & (train_patch_ds.patch_labels >= 0))[0]
    holdout_count = int(len(labeled_train_idx) * args.patch_eval_holdout_ratio)
    if len(labeled_train_idx) > 0 and holdout_count == 0 and args.patch_eval_holdout_ratio > 0:
        holdout_count = 1
    if holdout_count > 0:
        rng = np.random.RandomState(args.seed + 2026)
        patch_eval_holdout_indices = np.sort(rng.choice(labeled_train_idx, size=holdout_count, replace=False))
    else:
        patch_eval_holdout_indices = np.array([], dtype=np.int64)
    train_patch_indices = np.setdiff1d(all_train_indices, patch_eval_holdout_indices)
    print(
        f"[Patch Holdout] reserved={len(patch_eval_holdout_indices)} labeled patches "
        f"for patch-level evaluation, training_patches={len(train_patch_indices)}"
    )

    ssl_split = create_ssl_split_feature(
        train_patch_ds,
        labeled_slide_ratio=args.labeled_slide_ratio,
        labeled_patch_ratio=args.labeled_patch_ratio,
        seed=args.seed,
    )
    ssl_labeled_indices = np.setdiff1d(ssl_split["labeled_indices"], patch_eval_holdout_indices)
    ssl_unlabeled_indices = np.setdiff1d(ssl_split["unlabeled_indices"], patch_eval_holdout_indices)
    ssl_split["num_labeled"] = int(len(ssl_labeled_indices))
    ssl_split["num_unlabeled"] = int(len(ssl_unlabeled_indices))

    train_ssl_ds = SSLPatchFeatureDataset(
        train_patch_ds,
        ssl_labeled_indices,
        ssl_unlabeled_indices,
        strong_noise_std=args.strong_noise_std,
    )

    train_bag_loader = torch.utils.data.DataLoader(train_bag_ds, batch_size=1, shuffle=True, num_workers=args.num_workers)
    train_patch_subset = torch.utils.data.Subset(train_patch_ds, train_patch_indices.tolist())
    train_patch_loader = torch.utils.data.DataLoader(train_patch_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    train_ssl_loader = torch.utils.data.DataLoader(train_ssl_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_patch_loader = torch.utils.data.DataLoader(test_patch_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if len(patch_eval_holdout_indices) > 0:
        patch_eval_subset = torch.utils.data.Subset(train_patch_ds, patch_eval_holdout_indices.tolist())
        patch_eval_loader = torch.utils.data.DataLoader(patch_eval_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        patch_eval_loader = None

    projector = FeatureProjector(args.feature_input_dim, args.feature_dim).to(device)
    teacher_head = DSMILAttentionHead(args.feature_dim).to(device)
    student_head = StudentHead(args.feature_dim).to(device)
    ssl_classifier = SSLClassifier(args.feature_dim).to(device)

    opt_projector = torch.optim.AdamW(projector.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_teacher = torch.optim.AdamW(teacher_head.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_student = torch.optim.AdamW(student_head.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_ssl = torch.optim.AdamW(ssl_classifier.parameters(), lr=args.lr, weight_decay=1e-4)

    trainer = FeatureHybridTrainer(
        projector, teacher_head, student_head, ssl_classifier,
        opt_projector, opt_teacher, opt_student, opt_ssl,
        train_bag_loader, train_patch_loader, train_ssl_loader, test_patch_loader, patch_eval_loader,
        writer, device, args,
    )
    trainer.train()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(
        {
            "projector": projector.state_dict(),
            "teacher_head": teacher_head.state_dict(),
            "student_head": student_head.state_dict(),
            "ssl_classifier": ssl_classifier.state_dict(),
            "ssl_split": ssl_split,
            "args": vars(args),
        },
        os.path.join(args.save_dir, f"{exp_name}.pth"),
    )
    print(f"Training completed. Model saved to {os.path.join(args.save_dir, f'{exp_name}.pth')}")


if __name__ == "__main__":
    main()
