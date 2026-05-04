import argparse
import os.path as osp
import numpy as np
import torch
import mmcv

from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.cnn.utils import revert_sync_batchnorm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mmseg.models import build_segmentor
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import setup_multi_processes


# ===================== 配置 =====================
UNKNOWN_CLASS_ID = -1


def set_dataset_config(dataset_name):
    global UNKNOWN_CLASS_ID
    if dataset_name == 'ISPRS':
        UNKNOWN_CLASS_ID = 5
    elif dataset_name == 'LoveDA':
        UNKNOWN_CLASS_ID = 6


# ===================== 工具函数 =====================
def compute_unknown_ratio_from_eval_gt(gt_eval, unknown_class_id):
    """基于 eval label space 的 GT 计算 unknown ratio"""
    gt_eval = np.asarray(gt_eval)
    valid = gt_eval != 255
    if valid.sum() == 0:
        return np.nan
    return float(np.sum(gt_eval[valid] == unknown_class_id) / valid.sum())


def compute_per_image_miou(pred, gt, ignore_index=255, exclude_unknown=False, unknown_class_id=None):
    """
    计算单张图像的 mIoU
    pred: HxW, ndarray
    gt:   HxW, ndarray, eval label space
    """
    pred = np.asarray(pred).astype(np.int64)
    gt = np.asarray(gt).astype(np.int64)

    assert pred.shape == gt.shape, f"Shape mismatch: pred {pred.shape}, gt {gt.shape}"

    valid_mask = (gt != ignore_index)
    pred = pred[valid_mask]
    gt = gt[valid_mask]

    if gt.size == 0:
        return np.nan

    classes = np.union1d(np.unique(gt), np.unique(pred))

    if exclude_unknown:
        assert unknown_class_id is not None
        classes = classes[classes != unknown_class_id]

    ious = []
    for cls in classes:
        pred_c = (pred == cls)
        gt_c = (gt == cls)
        union = np.logical_or(pred_c, gt_c).sum()
        if union == 0:
            continue
        intersection = np.logical_and(pred_c, gt_c).sum()
        ious.append(intersection / union)

    if len(ious) == 0:
        return np.nan

    return float(np.mean(ious))


def get_sample_filename(dataset, idx):
    try:
        img_info = dataset.img_infos[idx]
        if isinstance(img_info, dict):
            if 'filename' in img_info:
                return osp.basename(img_info['filename'])
            if 'ann' in img_info and isinstance(img_info['ann'], dict) and 'seg_map' in img_info['ann']:
                return osp.basename(img_info['ann']['seg_map'])
    except Exception:
        pass
    return f'sample_{idx:04d}'


# ===================== 绘图函数 =====================
def plot_linear_regression_square(
        X_list,
        Y_list,
        save_path='image_level_analysis.png',
        dataset_name='Dataset'):
    """
    只绘制：
      - 散点图（论文风格）
      - 线性拟合曲线

    要求：
      1) 横坐标固定 [0, 1]
      2) 纵坐标为百分比 [0, 70]
      3) 超过 70% 的点忽略
      4) 正方形图
      5) 不显示 R^2
    """
    X = np.array(X_list, dtype=np.float64)
    Y = np.array(Y_list, dtype=np.float64)

    # 过滤非法值
    valid = ~(np.isnan(X) | np.isnan(Y) | np.isinf(X) | np.isinf(Y))
    X = X[valid]
    Y = Y[valid]

    if len(X) == 0:
        print('没有有效样本，无法绘图。')
        return

    # 转成百分比
    Y_percent = Y * 100.0

    # 只保留 <= 70% 的点
    keep = Y_percent <= 70.0
    X = X[keep]
    Y_percent = Y_percent[keep]

    if len(X) == 0:
        print('过滤掉 >70% 的样本后，没有有效样本，无法绘图。')
        return

    # 按横坐标升序排列
    sort_idx = np.argsort(X)
    X = X[sort_idx]
    Y_percent = Y_percent[sort_idx]

    # 线性拟合 y = ax + b
    coef = np.polyfit(X, Y_percent, deg=1)
    a, b = coef[0], coef[1]

    x_line = np.linspace(0.0, 1.0, 300)
    y_line = a * x_line + b

    # 论文风格绘图
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'STIXGeneral'],
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False
    })

    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    # 散点：更克制、更适合论文
    ax.scatter(
        X,
        Y_percent,
        s=28,
        c='#4C72B0',
        alpha=0.82,
        edgecolors='#FFFFFF',
        linewidths=0.6,
        marker='o',
        label='Samples',
        zorder=3
    )

    # 线性拟合线
    ax.plot(
        x_line,
        y_line,
        color='#C44E52',
        linewidth=2.0,
        linestyle='-',
        label='Linear fit',
        zorder=4
    )

    # 坐标范围固定
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 70.0)

    # 标签与标题
    ax.set_xlabel('Unknown Class Ratio', fontsize=13)
    ax.set_ylabel('mIoU (%)', fontsize=13)
    # ax.set_title(f'{dataset_name}', fontsize=14, pad=10)

    # 刻度
    ax.tick_params(axis='both', which='major', labelsize=11, direction='in', length=5, width=0.8)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.6)

    # 网格：轻量、论文风格
    ax.grid(True, which='major', linestyle='--', linewidth=0.55, color='#D9D9D9', alpha=0.8)
    ax.set_axisbelow(True)

    # 边框细化
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('#333333')

    # 图例
    ax.legend(
        loc='best',
        fontsize=11,
        frameon=True,
        fancybox=False,
        edgecolor='#B0B0B0'
    )

    # 正方形绘图区
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'\n✅ 图像已保存: {save_path}')
    print(f'过滤后有效样本数: {len(X)}')
    print(f'线性拟合方程: y = {a:.4f}x + {b:.4f}')


# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='LoveDA', choices=['ISPRS', 'LoveDA'])
    parser.add_argument('--config',
                        default='experiments/segformerb5/config_LoveDA/OSNet_40k_R2U.py',
                        help='test config path')
    parser.add_argument('--checkpoint',
                        default='/data/fywdata/fyw/UDA/OSUDA/MyNet/myresults_R2U_segformer/iter_40000.pth',
                        help='checkpoint path')
    parser.add_argument('--gpu-id', type=int, default=4)
    parser.add_argument('--save-path', type=str, default='image_level_analysis.png')
    parser.add_argument('--exclude-unknown-in-miou', action='store_true',
                        help='whether to exclude unknown class when computing per-image mIoU')
    args = parser.parse_args()

    set_dataset_config(args.dataset)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.gpu_ids = [args.gpu_id]
    setup_multi_processes(cfg)

    # dataset / dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    # model
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        model.PALETTE = dataset.PALETTE

    model = revert_sync_batchnorm(model)
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    model.eval()

    Xs, Ys = [], []

    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            pred = model(return_loss=False, rescale=True, **data)[0]
            pred = np.asarray(pred).astype(np.int64)

            gt = dataset.get_gt_seg_map_by_idx(idx)
            gt = np.asarray(gt).astype(np.int64)

            if pred.shape != gt.shape:
                print(f'Warning: shape mismatch at idx={idx}, pred={pred.shape}, gt={gt.shape}')
                continue

            unknown_ratio = compute_unknown_ratio_from_eval_gt(gt, UNKNOWN_CLASS_ID)

            miou = compute_per_image_miou(
                pred,
                gt,
                ignore_index=255,
                exclude_unknown=args.exclude_unknown_in_miou,
                unknown_class_id=UNKNOWN_CLASS_ID
            )

            Xs.append(unknown_ratio)
            Ys.append(miou)

            file_name = get_sample_filename(dataset, idx)
            print(
                f'[{idx+1:04d}/{len(dataset)}] {file_name} | '
                f'unknown ratio: {unknown_ratio:.4f} | '
                f'mIoU: {miou*100:.2f}%'
            )

    plot_linear_regression_square(
        Xs,
        Ys,
        save_path=args.save_path,
        dataset_name=args.dataset
    )


if __name__ == '__main__':
    main()