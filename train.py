import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
# 用于处理批量归一化的函数
from mmcv.cnn.utils import revert_sync_batchnorm
## 分布式训练相关的函数
from mmcv.runner import get_dist_info, init_dist
## 配置文件处理和版本信息获取
from mmcv.utils import Config, DictAction, get_git_hash
from mmseg import __version__
## 用于初始化随机种子和训练分割模型的函数
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
# 用于构建数据集的函数
from mmseg.datasets import build_dataset
## 用于构建分割模型的函数
from mmseg.models import build_segmentor
# # 环境信息收集、日志记录和多进程设置的函数
from mmseg.utils import collect_env, get_root_logger, setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config', default='experiments/deeplabv3/config/OSNet_40k_Potsdam2Vaihingen.py',
                        help='train config file path')
    # 添加工作目录参数，用于保存日志和模型
    parser.add_argument('--work-dir', default='./OSNet_R2U/', help='the dir to save logs and models')
    # 添加从检查点文件加载权重的参数
    parser.add_argument('--load-from', help='the checkpoint file to load weights from')
    # 添加从检查点文件恢复训练的参数
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    # 添加是否在训练过程中评估检查点的布尔参数
    parser.add_argument('--no-validate', action='store_true',
                        help='whether not to evaluate the checkpoint during training')
    # 创建一个互斥参数组，用于指定GPU的使用情况
    group_gpus = parser.add_mutually_exclusive_group()
    # 添加过时的GPU使用参数
    group_gpus.add_argument('--gpus', type=int,  # default=5,
                            help='(Deprecated, please use --gpu-id) number of gpus to use '
                                 '(only applicable to non-distributed training)')
    # 添加过时的GPU ID使用参数
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+',
                            help='(Deprecated, please use --gpu-id) ids of gpus to use '
                                 '(only applicable to non-distributed training)')
    # 添加GPU ID参数
    group_gpus.add_argument('--gpu-id', type=int, default=3, help='id of gpu to use '
                                                                  '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    # 添加设置CUDNN后端为确定性选项的布尔参数
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')
    # 添加覆盖配置文件设置的参数
    parser.add_argument('--options', nargs='+', action=DictAction, help='options')
    # 添加覆盖配置文件设置的新参数
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override some settings in the used config')
    # 添加作业启动器参数
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    # 添加本地排名参数，用于分布式训练
    parser.add_argument('--local_rank', type=int, default=0)
    # 添加自动从最新检查点恢复的布尔参数
    parser.add_argument('--auto-resume', action='store_true', help='resume from the latest checkpoint automatically.')
    args = parser.parse_args()
    # 设置环境变量LOCAL_RANK
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    # 如果同时指定了--options和--cfg-options，抛出异常
    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    # 如果指定了--options，发出警告并将其转换为--cfg-options
    if args.options:
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()
    # 从文件加载配置
    cfg = Config.fromfile(args.config)
    # 如果指定了--cfg-options，合并配置
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # 设置cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # 确定工作目录
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    # 如果指定了--load-from，设置加载权重的路径
    if args.load_from is not None:
        cfg.load_from = args.load_from
    # 如果指定了--resume-from，设置恢复训练的路径
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    # 处理GPU参数
    if args.gpus is not None:
        cfg.gpu_ids = range(1)

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]

    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]
    # 设置自动恢复
    cfg.auto_resume = args.auto_resume
    # 初始化分布式环境
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # 获取分布式信息
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # 创建工作目录
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # 保存配置文件
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # 初始化日志记录器
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    # 设置多进程
    setup_multi_processes(cfg)
    # 初始化元数据字典，用于记录重要信息
    meta = dict()
    # 记录环境信息
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    # 记录基本信息
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)
    # 构建分割模型
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # 如果不是分布式训练，处理SyncBN
    if not distributed:
        model = revert_sync_batchnorm(model)

    # 构建训练数据集
    datasets = [build_dataset(cfg.data.train)]

    # ===================== 核心修改：打印源域和目标域类别 =====================
    # 获取训练数据集实例（PVDataset_forAdap）
    train_dataset = datasets[0]

    # 打印源域信息
    logger.info("=" * 80)
    logger.info(f"【源域训练类别】: {train_dataset.source_included_classes}")
    logger.info(f"【源域有效类别数】: {train_dataset.source_num_classes}")
    logger.info(
        f"【源域被忽略的类别】: {[cls for cls in train_dataset.FULL_CLASSES if cls not in train_dataset.source_included_classes]}")
    logger.info(f"【源域标签映射表】: {train_dataset.source_label_map}")  # 查看原始标签→训练标签的映射

    # 打印目标域信息（目标域固定为全6类）
    logger.info(f"【目标域训练类别】: {train_dataset.FULL_CLASSES}")
    logger.info(f"【目标域类别数】: {len(train_dataset.FULL_CLASSES)}")
    logger.info("=" * 80)
    # ======================================================================

    # 如果工作流包括验证步骤，构建验证数据集
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)

        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    # 如果配置了检查点，保存mmseg版本、配置文件内容和类别名称
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # 为了方便可视化，添加类别属性
    model.CLASSES = datasets[0].CLASSES

    # 保存最佳检查点的元数据
    meta.update(cfg.checkpoint_config.meta)

    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()