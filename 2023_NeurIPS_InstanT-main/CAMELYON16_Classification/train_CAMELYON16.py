import argparse
import warnings
import os
import datetime
import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
from tensorboardX import SummaryWriter
# 只需要修改这一行import - 使用alias保持接口一致
from dataset_feat_CAMELYON16 import Camelyon16_feat_dataset as Lung_feat_clip, PatchSamplingDataset, prepare_data, \
    validate_dataset, args
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import dsmil


def cal_auc_multi_class(label, pred):
    """计算二分类的AUC"""
    if type(label) is not np.ndarray:
        label = label.detach().cpu().numpy()
    if type(pred) is not np.ndarray:
        pred = pred.detach().cpu().numpy()

    from sklearn.metrics import roc_auc_score
    try:
        assert len(label.shape) == 1, f"意外的标签形状: {label.shape}"
        assert pred.shape[1] == 2, f"意外的预测形状: {pred.shape}"
        auc = roc_auc_score(label, pred[:, 1])
        return [auc]
    except Exception as e:
        print(f"计算AUC时出错: {e}")
        print(f"标签形状: {label.shape}, 预测形状: {pred.shape}")
        print(f"唯一标签: {np.unique(label)}")
        return [0]


def train_one_epoch(epoch, model, loader, optimizer, dev, writer, args):
    model.train()

    # 如果数据集中有类别权重，则使用它们
    if hasattr(loader.dataset, 'dataset') and hasattr(loader.dataset.dataset, 'class_weights'):
        weights = loader.dataset.dataset.class_weights.to(dev)
    elif hasattr(loader.dataset, 'class_weights'):
        weights = loader.dataset.class_weights.to(dev)
    else:
        # 基于类别分布的默认权重
        labels = torch.tensor([label[1] for _, label, _ in loader.dataset.dataset])
        unique_labels, counts = torch.unique(labels, return_counts=True)
        weights = torch.tensor(len(labels)) / (len(unique_labels) * counts.float())

    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(dev))

    patch_label_gt = torch.zeros([loader.dataset.__len__()]).long()
    pred_prob_all = torch.zeros([loader.dataset.__len__(), 2])
    total_loss = 0

    for iter, (data, label, selected) in enumerate(tqdm(loader, desc=f'训练周期 {epoch}')):
        optimizer.zero_grad()
        niter = epoch * len(loader) + iter

        bag_label = label[1]
        patch_label_gt[selected] = bag_label

        if args.feature_noise > 0:
            data = data + torch.randn_like(data) * args.feature_noise

        # 确保数据是浮点型并在正确的设备上
        data = data.to(dev).float()
        if len(data.shape) == 3:  # 如果数据是3D的，移除额外的维度
            data = data.squeeze(0)
        elif len(data.shape) == 1:  # 如果数据是1D的，添加batch维度
            data = data.unsqueeze(0)
        bag_label = bag_label.to(dev).long()

        try:
            # 打印特征形状用于调试
            if iter == 0:
                print(f"特征形状: {data.shape}, 特征类型: {data.dtype}")

            # 前向传播
            ins_prediction, bag_prediction, attention, _ = model(data)
            max_prediction, _ = torch.max(ins_prediction, 0)

            bag_loss = criterion(bag_prediction, bag_label)
            max_loss = criterion(max_prediction.unsqueeze(0), bag_label)
            attention_reg = torch.mean(torch.sum(attention ** 2, dim=1))
            loss = bag_loss + args.max_weight * max_loss + args.attention_weight * attention_reg

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            final_prob = torch.softmax(bag_prediction, dim=1)
            pred_prob_all[selected, :] = final_prob.detach().cpu()

            if writer is not None:
                writer.add_scalar('train/loss', loss.item(), niter)
                writer.add_scalar('train/bag_loss', bag_loss.item(), niter)
                writer.add_scalar('train/max_loss', max_loss.item(), niter)
                writer.add_scalar('train/attention_reg', attention_reg.item(), niter)

            total_loss += loss.item()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f'警告: 在批次 {iter} 中内存不足')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                continue
            else:
                # 打印详细的错误信息
                print(f"数据形状: {data.shape}")
                print(f"错误: {e}")
                raise e

    # 计算指标
    mask = torch.any(pred_prob_all > 0, dim=1)  # 找到有效的预测
    if mask.sum() > 0:
        pred_logit = pred_prob_all[mask].argmax(1)
        patch_label_gt_valid = patch_label_gt[mask]
        train_acc = torch.sum(pred_logit == patch_label_gt_valid).float() / patch_label_gt_valid.shape[0]

        try:
            auc_all_cate = cal_auc_multi_class(patch_label_gt_valid, pred_prob_all[mask])
        except Exception as e:
            print(f"计算AUC时出错: {e}")
            auc_all_cate = [0]
    else:
        train_acc = torch.tensor(0.0)
        auc_all_cate = [0]

    if writer is not None:
        writer.add_scalar('train/epoch_acc', train_acc, epoch)
        writer.add_scalar('train/auc', auc_all_cate[0], epoch)
        writer.add_scalar('train/epoch_loss', total_loss / len(loader), epoch)

    return train_acc, auc_all_cate[0]


def evaluate(epoch, model, loader, dev, writer):
    model.eval()
    patch_label_gt = torch.zeros([loader.dataset.__len__()]).long()
    pred_prob_all = torch.zeros([loader.dataset.__len__(), 2])

    for iter, (data, label, selected) in enumerate(tqdm(loader, desc='评估中')):
        bag_label = label[1]
        patch_label_gt[selected] = bag_label

        try:
            predictions = []
            for _ in range(5):  # 测试时增强
                with torch.no_grad():
                    data_aug = data + torch.randn_like(data) * 0.01
                    data_aug = data_aug.to(dev).float()
                    if len(data_aug.shape) == 3:  # 如果数据是3D的，移除额外的维度
                        data_aug = data_aug.squeeze(0)
                    elif len(data_aug.shape) == 1:  # 如果数据是1D的，添加batch维度
                        data_aug = data_aug.unsqueeze(0)
                    ins_prediction, bag_prediction, attention, _ = model(data_aug)
                    final_prob = torch.softmax(bag_prediction, dim=1)
                    predictions.append(final_prob)

            final_prob = torch.mean(torch.stack(predictions), dim=0)
            pred_prob_all[selected, :] = final_prob.detach().cpu()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f'警告: 在批次 {iter} 中内存不足')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                continue
            else:
                print(f"评估错误: {e}")
                # 打印更多调试信息
                print(f"数据形状: {data.shape}")
                raise e

    # 计算指标
    mask = torch.any(pred_prob_all > 0, dim=1)  # 找到有效的预测
    if mask.sum() > 0:
        pred_logit = pred_prob_all[mask].argmax(1)
        patch_label_gt_valid = patch_label_gt[mask]
        val_acc = torch.sum(pred_logit == patch_label_gt_valid).float() / patch_label_gt_valid.shape[0]

        try:
            auc_all_cate = cal_auc_multi_class(patch_label_gt_valid, pred_prob_all[mask])
            kappa = cohen_kappa_score(patch_label_gt_valid, pred_logit)
            confusion_mat = confusion_matrix(patch_label_gt_valid, pred_logit)
        except Exception as e:
            print(f"计算评估指标时出错: {e}")
            auc_all_cate = [0]
            kappa = 0
            confusion_mat = np.zeros((2, 2))
    else:
        val_acc = torch.tensor(0.0)
        auc_all_cate = [0]
        kappa = 0
        confusion_mat = np.zeros((2, 2))

    print(f'\n周期 {epoch} 验证结果:')
    print(f'准确率: {val_acc:.4f}')
    print(f'AUC: {auc_all_cate[0]:.4f}')
    print(f'Kappa: {kappa:.4f}')
    print('混淆矩阵:')
    print(confusion_mat)

    if writer is not None:
        writer.add_scalar('val/acc', val_acc, epoch)
        writer.add_scalar('val/auc', auc_all_cate[0], epoch)
        writer.add_scalar('val/kappa', kappa, epoch)

    return val_acc, auc_all_cate[0]


def save_ckpt(state, save_name):
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    torch.save(state, save_name)


def get_parser():
    parser = argparse.ArgumentParser(description='Camelyon16分类 - Virchow2特征版本 (使用全部训练数据)')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--seed', default=500, type=int)
    parser.add_argument('--feats_size', default=2048, type=int,
                        help='特征维度，ResNet50为2048维，将自动检测')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--feature_noise', type=float, default=0.1)
    parser.add_argument('--max_weight', type=float, default=0.5)
    parser.add_argument('--attention_weight', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_patches', type=int, default=1000,
                        help='每个包的最大patch数量')
    parser.add_argument('--attention_batch_size', type=int, default=256,
                        help='注意力计算的批量大小')

    # 新增的Camelyon16特定参数
    parser.add_argument('--train_data_path', type=str,
                        default='/home/xiaoyuan/Data3/CAMELYON16/training_feature',
                        help='训练集特征数据路径')
    parser.add_argument('--test_data_path', type=str,
                        default='/home/xiaoyuan/Data3/CAMELYON16/testing_feature',
                        help='测试集特征数据路径')
    parser.add_argument('--train_wsi_limit', type=int, default=None,
                        help='训练集WSI数量限制（None表示使用所有数据）')

    args = parser.parse_args()
    return args


def main():
    args = get_parser()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # 更新全局默认参数以匹配命令行参数
    from dataset_feat_CAMELYON16 import args as dataset_args
    dataset_args.seed = args.seed
    dataset_args.max_patches = args.max_patches
    dataset_args.train_data_path = args.train_data_path
    dataset_args.test_data_path = args.test_data_path
    dataset_args.train_wsi_limit = args.train_wsi_limit

    # 准备数据
    train_dataset, test_dataset = prepare_data(dataset_args)

    # 检测特征维度
    if hasattr(train_dataset, 'dataset'):
        sample_feats = train_dataset.dataset.slide_feat_all[0]
    else:
        sample_feats = train_dataset.slide_feat_all[0]

    feature_dim = sample_feats.shape[1]
    print(f"检测到的特征维度: {feature_dim}")
    # 在args中更新特征维度
    args.feats_size = feature_dim

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    print(f"[数据] {len(train_loader.dataset)} 个训练样本")
    print(f"[数据] {len(test_loader.dataset)} 个测试样本")

    dev = f'cuda:{args.device}'

    # 使用检测到的特征维度初始化模型
    i_classifier = dsmil.FCLayer(in_size=args.feats_size).to(dev)
    b_classifier = dsmil.BClassifier(
        input_size=args.feats_size,
        output_class=2,
        dropout_v=args.dropout
    ).to(dev)
    model = dsmil.MILNet(i_classifier, b_classifier).to(dev)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.5, 0.9),
        weight_decay=args.weight_decay
    )

    # 加载checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        # 如果checkpoint中包含特征维度，更新args中的设置
        if 'feats_size' in checkpoint:
            args.feats_size = checkpoint['feats_size']
            print(f"从checkpoint加载特征维度: {args.feats_size}")
        print(f"从 {args.resume} 加载checkpoint")

    # 设置tensorboard - 修改为全数据集版本
    wsi_info = "all" if args.train_wsi_limit is None else str(args.train_wsi_limit)
    name = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S") + f"_Camelyon16_Bs{args.batch_size}_lr{args.lr}_feats{args.feats_size}_wsi{wsi_info}"
    writer = SummaryWriter(f'./runs_Camelyon16_DSMIL/{name}')

    # 训练循环
    best_val_auc = 0
    best_epoch = 0

    try:
        for epoch in range(start_epoch, args.epochs):
            print(f'\n周期 {epoch}/{args.epochs}')
            print('-' * 50)

            # 训练
            train_acc, train_auc = train_one_epoch(epoch, model, train_loader, optimizer, dev, writer, args)

            # 验证
            val_acc, val_auc = evaluate(epoch, model, test_loader, dev, writer)

            # 打印当前状态
            print('-' * 50)
            print(f'周期 {epoch}: 训练 AUC = {train_auc:.4f}, 验证 AUC = {val_auc:.4f}')
            print(f'目前最佳验证 AUC = {best_val_auc:.4f} 于周期 {best_epoch}')
            print('-' * 50)

            # 保存最佳模型
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch

                # 保存checkpoint - 修改为全数据集版本
                save_path = f"./checkpoints/Camelyon16_DSMIL/best_model_auc_{val_auc:.4f}.pth"
                save_ckpt({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_auc': best_val_auc,
                    'feats_size': args.feats_size,
                    'train_wsi_limit': args.train_wsi_limit,
                }, save_path)
                print(f"新的最佳模型已保存，验证 AUC: {val_auc:.4f}")

            # 每20个epoch保存一个checkpoint
            if epoch % 20 == 0 or epoch == args.epochs - 1:
                save_path = f"./checkpoints/Camelyon16_DSMIL/epoch_{epoch}_auc_{val_auc:.4f}.pth"
                save_ckpt({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_auc': val_auc,
                    'feats_size': args.feats_size,
                    'train_wsi_limit': args.train_wsi_limit,
                }, save_path)
                print(f"周期 {epoch} 模型已保存，验证 AUC: {val_auc:.4f}")

    except Exception as e:
        print(f"训练被错误中断: {e}")
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        # 保存当前模型作为断点
        save_path = f"./checkpoints/Camelyon16_DSMIL/interrupted_model_epoch_{epoch}.pth"
        save_ckpt({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_auc': best_val_auc,
            'feats_size': args.feats_size,
            'train_wsi_limit': args.train_wsi_limit,
        }, save_path)
        print(f"中断的模型已保存至 {save_path}")
        raise e

    writer.close()
    print(f"\n训练完成. 最佳验证 AUC: {best_val_auc:.4f} 于周期 {best_epoch}")

    # 保存最终模型
    final_save_path = f"./checkpoints/Camelyon16_DSMIL/final_model_auc_{best_val_auc:.4f}.pth"
    save_ckpt({
        'epoch': best_epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_val_auc': best_val_auc,
        'feats_size': args.feats_size,
        'train_wsi_limit': args.train_wsi_limit,
    }, final_save_path)
    print(f"最终模型已保存至 {final_save_path}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()