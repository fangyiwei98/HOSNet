# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import HOOKS, build_optimizer, build_runner, get_dist_info, OptimizerHook
from mmcv.utils import build_from_cfg

from mmseg import digit_version
from mmseg.core import DistEvalHook, EvalHook
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import find_latest_checkpoint, get_root_logger
from mmseg.core.optimizer import build_optimizers as build_optimizers_multi
from mmseg.core.ddp_wrapper import DistributedDataParallelWrapper

def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]
    

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        use_ddp_wrapper = cfg.get('use_ddp_wrapper', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        if use_ddp_wrapper:
            mmcv.print_log('Use DDP Wrapper.', 'mmgen')
            model = DistributedDataParallelWrapper(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    # 构建Runner，用于控制训练过程
    '''
    根据配置为模型的不同部分（如主网络结构和子网络）构建优化器。
    根据配置决定是否以及如何配置优化器钩子。
    '''
    if isinstance(cfg.optimizer, dict) and cfg.optimizer.get('backbone', None):
        optimizer = build_optimizers_multi(model, cfg.optimizer)
        if cfg.get('optimizer_cfg', None) is None:
            cfg.optimizer_config = None
        # default to use OptimizerHook
        elif distributed and 'type' not in cfg.optimizer_config:
            cfg.optimizer_config = OptimizerHook(**cfg.optimizer_config)
        else:
            cfg.optimizer_config = cfg.optimizer_config
    elif isinstance(cfg.optimizer, dict) and cfg.optimizer.get('backbone_s', None):
        optimizer = build_optimizers_multi(model, cfg.optimizer)
        if cfg.get('optimizer_cfg', None) is None:
            cfg.optimizer_config = None
        # default to use OptimizerHook
        elif distributed and 'type' not in cfg.optimizer_config:
            cfg.optimizer_config = OptimizerHook(**cfg.optimizer_config)
        else:
            cfg.optimizer_config = cfg.optimizer_config
    else:
        optimizer = build_optimizer(model, cfg.optimizer)

    # 如果配置中没有指定Runner，则使用默认设置
    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # 注册训练钩子（hooks），用于在训练过程中执行特定操作（如学习率调整、保存检查点等）
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # 设置时间戳，以确保日志和检查点文件名一致
    runner.timestamp = timestamp

    # 如果需要在训练过程中进行验证，则注册验证钩子，并构建验证数据集和加载器
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        ## For adapseg trainval
        if cfg.runner['type'] == 'DynamicIterBasedRunner':
            eval_cfg['by_epoch'] = False
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')   
    # 如果配置中定义了自定义钩子，则注册这些钩子
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    # 如果指定了从某个检查点恢复训练或自动恢复，则加载相应的检查点
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
        if resume_from is not None:
            cfg.resume_from = resume_from
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    # 运行Runner，开始训练过程
    if cfg.get('total_iters', None):
        runner.run(data_loaders, cfg.workflow, cfg.total_iters)
    else:
        runner.run(data_loaders, cfg.workflow)