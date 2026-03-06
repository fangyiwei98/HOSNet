# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time

import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg import digit_version
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description='mmseg test (and eval) a model')
    # 添加测试配置文件和检查点文件的参数
    parser.add_argument('--config', default='experiments/segformerb5/config/PCANet_40k_PotsdamRGB2Vaihingen.py', help='test config file path')
    parser.add_argument('--checkpoint', default='/data/fywdata/code/fyw/UDA/APANet/experiments/segformerb5/myresults_P(RGB)2V/iter_37000.pth', help='checkpoint file')
    # 添加工作目录参数，用于保存评估结果
    parser.add_argument('--work-dir',help=('if specified, the evaluation metric results will be dumped into the directory as json'))
    # 添加使用翻转和多尺度增强的参数
    parser.add_argument('--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    # 添加输出结果文件的参数
    parser.add_argument('--out', help='output result file in pickle format')
    # 添加仅格式化输出结果而不执行评估的参数
    parser.add_argument('--format-only',action='store_true',help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    # 添加评估指标的参数
    parser.add_argument('--eval',default='mIoU', type=str,nargs='+',help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    # 添加显示结果的参数
    parser.add_argument('--show', action='store_true', help='show results')
    # 添加保存绘制图像的目录参数
    parser.add_argument('--show-dir',default='/data/fywdata/code/fyw/UDA/APANet/experiments/segformerb5/myresults_P(RGB)2V/vis/iter_37000/', help='directory where painted images will be saved')
    # 添加使用GPU收集结果的参数
    parser.add_argument('--gpu-collect',action='store_true',help='whether to use gpu to collect results.')
    # 添加指定GPU ID的参数
    parser.add_argument('--gpu-id',type=int,default=3,help='id of gpu to use (only applicable to non-distributed testing)')
    # 添加临时目录参数，用于收集多个工作进程的结果
    parser.add_argument('--tmpdir',help='tmp directory used for collecting results from multiple workers, available when gpu_collect is not specified')
    # 添加覆盖配置文件设置的参数
    parser.add_argument('--options',nargs='+',action=DictAction,help='none.')
    # 添加覆盖配置文件设置的新参数
    parser.add_argument('--cfg-options',nargs='+',action=DictAction,help='none.')
    # 添加评估自定义选项的参数
    parser.add_argument('--eval-options',nargs='+',action=DictAction,help='custom options for evaluation')
    # 添加作业启动器参数
    parser.add_argument('--launcher',choices=['none', 'pytorch', 'slurm', 'mpi'],default='none',help='job launcher')
    # 添加绘制分割图的不透明度参数
    parser.add_argument('--opacity',type=float,default=1,help='Opacity of painted segmentation map. In (0, 1] range.')
    # 添加本地排名参数，用于分布式训练
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    # 设置环境变量LOCAL_RANK
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        args.cfg_options = args.options
    return args


def main():
    args = parse_args()
    # 确保至少指定了一个操作（保存/评估/格式化/显示结果）
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')
    # 如果同时指定了--eval和--format-only，抛出异常
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')
    # 确保输出文件是pkl文件
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    # 从文件加载配置
    cfg = mmcv.Config.fromfile(args.config)
    # 如果指定了--cfg-options，合并配置
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # 如果启用了增强测试，设置数据增强的参数
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    # 设置模型为测试模式
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    # 如果指定了GPU ID，设置配置
    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    # 初始化分布式环境
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
        if len(cfg.gpu_ids) > 1:
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    # 获取分布式信息
    rank, _ = get_dist_info()
    # 如果工作目录被指定且是主进程，则创建工作目录并设置时间戳
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(args.work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(args.work_dir,
                                 f'eval_single_scale_{timestamp}.json')
    elif rank == 0:
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(work_dir,
                                 f'eval_single_scale_{timestamp}.json')

    # 构建数据加载器
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # 构建模型并加载检查点
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # 从检查点中获取类别和调色板信息
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    ## 设置评估参数
    eval_kwargs = {} if args.eval_options is None else args.eval_options


    # 评估是否在格式化结果上执行
    eval_on_format_results = (
        args.eval is not None and 'cityscapes' in args.eval)
    if eval_on_format_results:
        assert len(args.eval) == 1, 'eval on format results is not ' \
                                    'applicable for metrics other than ' \
                                    'cityscapes'
    if args.format_only or eval_on_format_results:
        if 'imgfile_prefix' in eval_kwargs:
            tmpdir = eval_kwargs['imgfile_prefix']
        else:
            tmpdir = '.format_cityscapes'
            eval_kwargs.setdefault('imgfile_prefix', tmpdir)
        mmcv.mkdir_or_exist(tmpdir)
    else:
        tmpdir = None
    #不是分布式训练
    if not distributed:
        if not torch.cuda.is_available():
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        model = revert_sync_batchnorm(model)
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        results = single_gpu_test(
            model,
            data_loader,
            args.show,
            args.show_dir,
            False,
            args.opacity,
            pre_eval=args.eval is not None and not eval_on_format_results,
            format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        results = multi_gpu_test(
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
            False,
            pre_eval=args.eval is not None and not eval_on_format_results,
            format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(results, args.out)
        if args.eval:
            eval_kwargs.update(metric=args.eval)
            metric = dataset.evaluate(results, **eval_kwargs)
            metric_dict = dict(config=args.config, metric=metric)
            mmcv.dump(metric_dict, json_file, indent=4)
            if tmpdir is not None and eval_on_format_results:
                # remove tmp dir when cityscapes evaluation
                shutil.rmtree(tmpdir)


if __name__ == '__main__':
    main()
