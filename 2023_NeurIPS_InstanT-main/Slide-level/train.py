# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import logging
import random
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from semilearn.algorithms import get_algorithm, name2alg
from semilearn.imb_algorithms import get_imb_algorithm, name2imbalg
from semilearn.core.utils import get_net_builder, get_logger, get_port, send_model_cuda, count_parameters, \
    over_write_args_from_file, TBLog

# 添加AUC计算支持
from sklearn.metrics import roc_auc_score

# 导入slide聚合工具
from slide_aggregation_utils import SlideAggregator, compute_slide_level_metrics, save_slide_results


def get_config():
    from semilearn.algorithms.utils import str2bool

    parser = argparse.ArgumentParser(description='Semi-Supervised Learning (USB)')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str, default='fixmatch')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str)
    parser.add_argument('-o', '--overwrite', action='store_true', default=True)
    parser.add_argument('--use_tensorboard', action='store_true', help='Use tensorboard to plot and save curves')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb to plot and save curves')
    parser.add_argument('--use_aim', action='store_true', help='Use aim to plot and save curves')

    '''
    Training Configuration of FixMatch
    '''
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=20,
                        help='total number of training iterations')
    parser.add_argument('--num_warmup_iter', type=int, default=0,
                        help='cosine linear warmup iterations')
    parser.add_argument('--num_eval_iter', type=int, default=10,
                        help='evaluation frequency')
    parser.add_argument('--num_log_iter', type=int, default=5,
                        help='logging frequency')
    parser.add_argument('-nl', '--num_labels', type=int, default=400)
    parser.add_argument('-bsz', '--batch_size', type=int, default=8)
    parser.add_argument('--uratio', type=int, default=1,
                        help='the ratio of unlabeled data to labeled data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)

    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--layer_decay', type=float, default=1.0,
                        help='layer-wise learning rate decay, default to 1.0 which means no layer decay')

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='wrn_28_2')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--use_pretrain', default=False, type=str2bool)
    parser.add_argument('--pretrain_path', default='', type=str)

    '''
    Algorithms Configurations
    '''

    ## core algorithm setting
    parser.add_argument('-alg', '--algorithm', type=str, default='fixmatch', help='ssl algorithm')
    parser.add_argument('--use_cat', type=str2bool, default=True, help='use cat operation in algorithms')
    parser.add_argument('--amp', type=str2bool, default=False, help='use mixed precision training or not')
    parser.add_argument('--clip_grad', type=float, default=0)

    ## imbalance algorithm setting
    parser.add_argument('-imb_alg', '--imb_algorithm', type=str, default=None, help='imbalance ssl algorithm')

    '''
    Data Configurations
    '''

    ## standard setting configurations
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('-ds', '--dataset', type=str, default='cifar10')
    parser.add_argument('-nc', '--num_classes', type=int, default=10)
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--include_lb_to_ulb', type=str2bool, default='True',
                        help='flag of including labeled data into unlabeled data, default to True')

    ## imbalanced setting arguments
    parser.add_argument('--lb_imb_ratio', type=int, default=1, help="imbalance ratio of labeled data, default to 1")
    parser.add_argument('--ulb_imb_ratio', type=int, default=1, help="imbalance ratio of unlabeled data, default to 1")
    parser.add_argument('--ulb_num_labels', type=int, default=None,
                        help="number of labels for unlabeled data, used for determining the maximum number of labels in imbalanced setting")

    ## cv dataset arguments
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--crop_ratio', type=float, default=0.875)

    ## nlp dataset algorithms
    parser.add_argument('--max_length', type=int, default=512)

    ## speech dataset algorithms
    parser.add_argument('--max_length_seconds', type=float, default=4.0)
    parser.add_argument('--sample_rate', type=int, default=16000)

    '''
    multi-GPUs & Distributed Training
    '''

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('-du', '--dist-url', default='tcp://127.0.0.1:11111', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=str2bool, default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    ## slide aggregation arguments
    parser.add_argument('--slide_aggregation_method', type=str, default='max',
                        choices=['max', 'mean', 'top_k'],
                        help='method for aggregating patch predictions to slide level')
    parser.add_argument('--save_slide_results', action='store_true',
                        help='save detailed slide-level prediction results')

    # config file
    parser.add_argument('--c', type=str, default='')

    # add algorithm specific parameters
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    for argument in name2alg[args.algorithm].get_argument():
        parser.add_argument(argument.name, type=argument.type, default=argument.default, help=argument.help)

    # add imbalanced algorithm specific parameters
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    if args.imb_algorithm is not None:
        for argument in name2imbalg[args.imb_algorithm].get_argument():
            parser.add_argument(argument.name, type=argument.type, default=argument.default, help=argument.help)
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    return args


def main(args):
    '''
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    '''

    assert args.num_train_iter % args.epoch == 0, \
        f"# total training iter. {args.num_train_iter} is not divisible by # epochs {args.epoch}"

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and args.overwrite and args.resume == False:
        import shutil
        shutil.rmtree(save_path)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading paths are same. \
                            If you want over-write, give --overwrite in the argument.')

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu == 'None':
        args.gpu = None
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()  # number of gpus of each node

    if args.multiprocessing_distributed:
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size

        # args=(,) means the arguments of main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def compute_patch_and_slide_auc(model, eval_loader, args, logger):
    """
    计算patch级别和slide级别的AUC
    """
    eval_model = model.ema_model if hasattr(model, 'ema_model') and model.ema_model is not None else model.model
    eval_model.eval()

    # 初始化slide聚合器
    slide_aggregator = SlideAggregator(
        aggregation_method=args.slide_aggregation_method,
        logger=logger
    )

    all_patch_labels = []
    all_patch_probs = []
    device = torch.device(f'cuda:{args.gpu}' if args.gpu is not None else 'cpu')

    try:
        with torch.no_grad():
            for batch in eval_loader:
                if isinstance(batch, dict):
                    inputs = batch.get('x_lb', batch.get('x', batch.get('image'))).to(device)
                    labels = batch.get('y_lb', batch.get('y', batch.get('target', batch.get('label')))).to(device)
                    # 尝试获取路径信息
                    paths = batch.get('paths', None)
                else:
                    inputs, labels = batch[:2]
                    inputs, labels = inputs.to(device), labels.to(device)
                    paths = batch[2] if len(batch) > 2 else None

                outputs = eval_model(inputs)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                probs = torch.softmax(logits, dim=1)

                # 累积patch级别数据
                all_patch_labels.append(labels)
                all_patch_probs.append(probs)

                # 如果有路径信息，添加到slide聚合器
                if paths is not None:
                    # 从数据集中获取路径信息
                    try:
                        if hasattr(eval_loader.dataset, 'paths'):
                            # 假设batch中的索引可以用来获取路径
                            batch_paths = []
                            for i in range(inputs.size(0)):
                                if hasattr(eval_loader.dataset, 'paths'):
                                    # 这里需要根据实际的数据集结构调整
                                    idx = (len(all_patch_labels) - 1) * eval_loader.batch_size + i
                                    if idx < len(eval_loader.dataset.paths):
                                        batch_paths.append(eval_loader.dataset.paths[idx])

                            if batch_paths:
                                slide_aggregator.add_batch(probs, labels, batch_paths)
                    except Exception as e:
                        logger.warning(f"无法提取路径信息进行slide聚合: {e}")

        if not all_patch_labels:
            logger.warning("评估数据为空")
            return None, None

        # 计算patch级别指标
        all_patch_labels = torch.cat(all_patch_labels)
        all_patch_probs = torch.cat(all_patch_probs)

        # 过滤有效标签
        valid_mask = all_patch_labels >= 0
        if valid_mask.sum() == 0:
            logger.warning("没有有效的标签")
            return None, None

        valid_labels = all_patch_labels[valid_mask].cpu().numpy()
        valid_probs = all_patch_probs[valid_mask].cpu().numpy()

        # 计算patch级别准确率
        pred_labels = np.argmax(valid_probs, axis=1)
        patch_accuracy = (pred_labels == valid_labels).mean()

        # 计算patch级别AUC
        patch_auc = None
        try:
            unique_labels = np.unique(valid_labels)
            if len(unique_labels) == 2 and set(unique_labels).issubset({0, 1}):
                y_score = valid_probs[:, 1] if valid_probs.shape[1] == 2 else valid_probs[:, 0]
                patch_auc = roc_auc_score(valid_labels, y_score)
                logger.info(f"Patch级别 - 准确率: {patch_accuracy:.4f}, AUC: {patch_auc:.4f}")
        except Exception as e:
            logger.warning(f"Patch AUC计算失败: {e}")

        # 计算slide级别指标
        slide_results = slide_aggregator.compute_slide_metrics()
        slide_auc = None

        if slide_results and slide_results['metrics']:
            slide_metrics = slide_results['metrics']
            slide_auc = slide_metrics.get('auc')
            slide_accuracy = slide_metrics.get('accuracy')

            logger.info(f"Slide级别 - 准确率: {slide_accuracy:.4f}")
            if slide_auc is not None:
                logger.info(f"Slide级别 - AUC: {slide_auc:.4f}")
                logger.info(f"聚合方法: {args.slide_aggregation_method}")
                logger.info(f"slide数量: {slide_metrics['num_slides']}")

            # 保存详细结果
            if args.save_slide_results and slide_results:
                save_path = os.path.join(args.save_dir, args.save_name, 'slide_results.json')
                save_slide_results(
                    slide_results['slide_probs'],
                    slide_results['slide_labels'],
                    slide_results['slide_names'],
                    slide_results['aggregation_info'],
                    save_path
                )
        else:
            logger.warning("无法计算slide级别指标 - 可能缺少路径信息")

        return patch_auc, slide_auc

    except Exception as e:
        logger.error(f"评估过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    finally:
        eval_model.train()


def main_worker(gpu, ngpus_per_node, args):
    '''
    main_worker is conducted on each GPU.
    '''

    global best_acc1
    args.gpu = gpu

    # random seed has to be set for the synchronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # SET UP FOR DISTRIBUTED TRAINING
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu  # compute global rank

        # set distributed group:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None
    if args.rank % ngpus_per_node == 0:
        tb_log = TBLog(save_path, 'tensorboard', use_tensorboard=args.use_tensorboard)
        logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.info(f"Use GPU: {args.gpu} for training")

    _net_builder = get_net_builder(args.net, args.net_from_name)
    # optimizer, scheduler, datasets, dataloaders with be set in algorithms
    if args.imb_algorithm is not None:
        model = get_imb_algorithm(args, _net_builder, tb_log, logger)
    else:
        model = get_algorithm(args, _net_builder, tb_log, logger)
    logger.info(f'Number of Trainable Params: {count_parameters(model.model)}')

    # 为CAMELYON16数据集添加AUC评估和slide级别聚合
    if args.dataset == 'camelyon16':
        logger.info("启用CAMELYON16数据集AUC计算和slide级别聚合")
        logger.info(f"Slide聚合方法: {args.slide_aggregation_method}")

        # 保存原始的train方法
        original_train = model.train

        def enhanced_train():
            """增强的训练方法，包含slide级别评估"""
            # 执行原始训练
            original_train()

            # 训练完成后计算最终的patch和slide级别指标
            logger.info("计算最终patch和slide级别指标...")

            if hasattr(model, 'loader_dict') and 'eval' in model.loader_dict:
                patch_auc, slide_auc = compute_patch_and_slide_auc(
                    model, model.loader_dict['eval'], args, logger
                )

                # 更新结果字典
                if patch_auc is not None:
                    model.results_dict['eval_patch_auc'] = patch_auc
                    model.results_dict['eval_auc'] = patch_auc  # 保持兼容性
                    model.results_dict['patch_auc'] = patch_auc
                    logger.info(f"Final result - patch_auc: {patch_auc:.4f}")

                if slide_auc is not None:
                    model.results_dict['eval_slide_auc'] = slide_auc
                    model.results_dict['slide_auc'] = slide_auc
                    logger.info(f"Final result - slide_auc: {slide_auc:.4f}")
                    logger.info(f"Final result - slide_aggregation_method: {args.slide_aggregation_method}")

                    # 输出关键信息
                    logger.info(f"🎯 关键结果:")
                    logger.info(f"   Patch级别AUC: {patch_auc:.4f}")
                    logger.info(f"   Slide级别AUC: {slide_auc:.4f} (聚合方法: {args.slide_aggregation_method})")
                else:
                    logger.warning("未能计算slide级别AUC - 请检查数据集路径信息")
            else:
                logger.warning("未找到评估数据加载器，跳过最终AUC计算")

        # 替换训练方法
        model.train = enhanced_train

    # SET Devices for (Distributed) DataParallel
    model.model = send_model_cuda(args, model.model)
    model.ema_model = send_model_cuda(args, model.ema_model, clip_batch=False)
    logger.info(f"Arguments: {model.args}")

    # If args.resume, load checkpoints from args.load_path
    if args.resume and os.path.exists(args.load_path):
        try:
            model.load_model(args.load_path)
        except:
            logger.info("Fail to resume load path {}".format(args.load_path))
            args.resume = False
    else:
        logger.info("Resume load path {} does not exist".format(args.load_path))

    if hasattr(model, 'warmup'):
        logger.info(("Warmup stage"))
        model.warmup()

    # START TRAINING of FixMatch
    logger.info("Model training")
    model.train()

    # print validation (and test results)
    for key, item in model.results_dict.items():
        logger.info(f"Model result - {key} : {item}")

    if hasattr(model, 'finetune'):
        logger.info("Finetune stage")
        model.finetune()

    logging.warning(f"GPU {args.rank} training is FINISHED")


if __name__ == "__main__":
    args = get_config()
    port = get_port()
    args.dist_url = "tcp://127.0.0.1:" + str(port)
    main(args)