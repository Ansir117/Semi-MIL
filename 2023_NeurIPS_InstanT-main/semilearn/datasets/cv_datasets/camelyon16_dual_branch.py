import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, DistAlignEMAHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.core.hooks import Hook


class AttentionBagHead(nn.Module):
    """Teacher: Attention-based MIL"""

    def __init__(self, input_dim=2048, hidden_dim=512, num_classes=2):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """x: [N, feat_dim] - è¾“å…¥å¿…é¡»æ˜¯float32"""
        A = self.attention(x)  # [N, 1]
        A = torch.softmax(A, dim=0)  # normalize
        bag_feat = torch.sum(x * A, dim=0, keepdim=True)  # [1, feat_dim]
        logits = self.classifier(bag_feat)  # [1, num_classes]
        return logits.squeeze(0), A.squeeze(1)


class InstanceHead(nn.Module):
    """Student: Instance classifier"""

    def __init__(self, input_dim=2048, hidden_dim=512, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """x: [N, feat_dim] - è¾“å…¥å¿…é¡»æ˜¯float32"""
        return self.fc(x)


class DualBranchThresholdingHook(Hook):
    def __init__(self, num_classes, p_cutoff=0.95):
        super().__init__()
        self.p_cutoff = p_cutoff

    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb=None, **kwargs):
        if logits_x_ulb is None:
            return None
        probs = torch.softmax(logits_x_ulb, dim=-1)
        max_probs = probs.max(dim=-1)[0]
        return max_probs.ge(self.p_cutoff).float()


@ALGORITHMS.register('instant_dual')
class InstanTDual(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        # å‚æ•°è®¾ç½®
        self.p_cutoff = getattr(args, 'p_cutoff', 0.95)
        self.T = getattr(args, 'T', 0.5)
        self.use_hard_label = getattr(args, 'hard_label', True)
        self.ema_p = getattr(args, 'ema_p', 0.999)
        self.teacher_warmup = getattr(args, 'teacher_warmup_iter', 5000)
        self.student_period = getattr(args, 'student_update_period', 2)
        self.neg_weight = getattr(args, 'stu_loss_weight_neg', 0.1)

        super().__init__(args, net_builder, tb_log, logger)

        # åˆ›å»ºteacherå’Œstudent heads (ç¡®ä¿åœ¨æ­£ç¡®çš„è®¾å¤‡ä¸Š)
        feat_dim = 2048 if 'resnet50' in str(self.model).lower() else 512
        self.teacher_head = AttentionBagHead(feat_dim, 256, self.num_classes).cuda(self.gpu)
        self.student_head = InstanceHead(feat_dim, 256, self.num_classes).cuda(self.gpu)

        # ä¼˜åŒ–å™¨
        self.teacher_opt = torch.optim.SGD(self.teacher_head.parameters(),
                                           lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        self.student_opt = torch.optim.SGD(self.student_head.parameters(),
                                           lr=self.args.lr, momentum=0.9, weight_decay=5e-4)

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, momentum=self.ema_p, p_target_type='model'),
            "DistAlignHook"
        )
        self.register_hook(
            DualBranchThresholdingHook(num_classes=self.num_classes, p_cutoff=self.p_cutoff),
            "MaskingHook"
        )
        super().set_hooks()

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # ===== Phase 1: Patch-level SSL =====
        with self.amp_cm():
            # Forward
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb)
                logits_x_lb, feats_x_lb = outs_x_lb['logits'], outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s, feats_x_ulb_s = outs_x_ulb_s['logits'], outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w, feats_x_ulb_w = outs_x_ulb_w['logits'], outs_x_ulb_w['feat']

            # Patch-level supervised loss
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # Pseudo-labeling
            probs_x_lb = self.compute_prob(logits_x_lb.detach())
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook",
                                           probs_x_ulb=probs_x_ulb_w, probs_x_lb=probs_x_lb)
            mask = self.call_hook("masking", "MaskingHook",
                                  logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          logits=probs_x_ulb_w, use_hard_label=self.use_hard_label,
                                          T=self.T, softmax=False)

            # Patch-level unsupervised loss
            unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss

        # Backward (åªä¼˜åŒ–backbone)
        out_dict = self.process_out_dict(loss=total_loss, feat={'x_lb': feats_x_lb})

        # ===== Phase 2: WENO-style Teacher-Student =====
        teacher_loss = torch.tensor(0.0).cuda(self.gpu)
        student_loss = torch.tensor(0.0).cuda(self.gpu)

        if self.it >= self.teacher_warmup:
            # Teacher: ç”¨æ‰€æœ‰labeled patchesä½œä¸ºä¸€ä¸ªbagè®­ç»ƒ
            # å…³é”®ä¿®å¤ï¼šè½¬æ¢ä¸ºfloat32é¿å…dtypeä¸åŒ¹é…
            with torch.no_grad():
                feats_all = feats_x_lb.detach().float()

            bag_logits, attention = self.teacher_head(feats_all)
            bag_label = (y_lb.max() > 0).long()  # bag label = 1 if any positive patch
            teacher_loss = F.cross_entropy(bag_logits.unsqueeze(0), bag_label.unsqueeze(0))

            self.teacher_opt.zero_grad()
            teacher_loss.backward()
            self.teacher_opt.step()

            # Student: æ¯Nä¸ªiterationæ›´æ–°ä¸€æ¬¡
            if self.it % self.student_period == 0:
                # å…³é”®ä¿®å¤ï¼šè½¬æ¢ä¸ºfloat32
                with torch.no_grad():
                    _, attention_normed = self.teacher_head(feats_x_lb.detach().float())
                    # è´Ÿæ ·æœ¬patchesçš„pseudo labelè®¾ä¸º0
                    pseudo_labels = attention_normed.clone()
                    pseudo_labels[y_lb == 0] = 0

                # Student forward (å…³é”®ä¿®å¤ï¼šè½¬æ¢ä¸ºfloat32)
                student_logits = self.student_head(feats_x_lb.detach().float())
                student_probs = torch.softmax(student_logits, dim=1)

                # åŠ æƒloss (WENO style)
                loss_neg = -self.neg_weight * (1 - pseudo_labels) * torch.log(student_probs[:, 0] + 1e-8)
                loss_pos = -(1 - self.neg_weight) * pseudo_labels * torch.log(student_probs[:, 1] + 1e-8)
                student_loss = (loss_neg + loss_pos).mean()

                self.student_opt.zero_grad()
                student_loss.backward()
                self.student_opt.step()

        log_dict = self.process_log_dict(
            sup_loss=sup_loss.item(),
            unsup_loss=unsup_loss.item(),
            teacher_loss=teacher_loss.item() if isinstance(teacher_loss, torch.Tensor) else 0.0,
            student_loss=student_loss.item() if isinstance(student_loss, torch.Tensor) else 0.0,
            total_loss=total_loss.item(),
            util_ratio=mask.float().mean().item()
        )

        return out_dict, log_dict

    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        """æ·»åŠ slide-levelè¯„ä¼°"""
        # å…ˆåšpatch-levelè¯„ä¼°
        result = super().evaluate(eval_dest, out_key, return_logits)

        # å†åšslide-levelè¯„ä¼°
        if hasattr(self, 'teacher_head'):
            slide_acc = self._eval_slide_level(eval_dest)
            if slide_acc is not None:
                result['slide_acc'] = slide_acc
                self.print_fn(f"Slide accuracy: {slide_acc:.4f}")

        return result

    def _eval_slide_level(self, eval_dest='eval'):
        """ç®€å•çš„slideè¯„ä¼°ï¼šæ¯batchå½“ä½œä¸€ä¸ªbag"""
        self.model.eval()
        self.teacher_head.eval()

        loader = self.loader_dict[eval_dest]
        correct, total = 0, 0

        with torch.no_grad():
            for data in loader:
                if isinstance(data, dict):
                    x = data.get('x_lb', data.get('x')).cuda(self.gpu)
                    y = data.get('y_lb', data.get('y')).cuda(self.gpu)
                else:
                    x, y = data[0].cuda(self.gpu), data[1].cuda(self.gpu)

                # Extract features
                outputs = self.model(x)
                feats = outputs['feat'] if isinstance(outputs, dict) else outputs
                # å…³é”®ä¿®å¤ï¼šç¡®ä¿æ˜¯float32
                feats = feats.float()

                # Bag-level prediction
                bag_logits, _ = self.teacher_head(feats)
                bag_pred = bag_logits.argmax()
                bag_label = (y.max() > 0).long()

                if bag_pred == bag_label:
                    correct += 1
                total += 1

        self.model.train()
        self.teacher_head.train()

        return correct / total if total > 0 else 0

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict.update({
            'teacher_head': self.teacher_head.state_dict(),
            'student_head': self.student_head.state_dict(),
            'teacher_opt': self.teacher_opt.state_dict(),
            'student_opt': self.student_opt.state_dict(),
        })
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        if 'teacher_head' in checkpoint:
            self.teacher_head.load_state_dict(checkpoint['teacher_head'])
            self.student_head.load_state_dict(checkpoint['student_head'])
            self.teacher_opt.load_state_dict(checkpoint['teacher_opt'])
            self.student_opt.load_state_dict(checkpoint['student_opt'])
            self.print_fn("Dual-branch loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--teacher_warmup_iter', int, 5000),
            SSL_Argument('--student_update_period', int, 2),
            SSL_Argument('--stu_loss_weight_neg', float, 0.1),
        ]