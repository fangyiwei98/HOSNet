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
def compute_unknown_ratio_from_eval_gt_multi(gt_eval, unknown_class_ids):
    gt_eval = np.asarray(gt_eval)
    valid = gt_eval != 255
    if valid.sum() == 0:
        return np.nan
    unknown_mask = np.isin(gt_eval[valid], unknown_class_ids)
    return float(np.sum(unknown_mask) / valid.sum())


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


def build_model_and_loader(config_path, checkpoint_path, gpu_id):
    cfg = mmcv.Config.fromfile(config_path)
    cfg.data.test.test_mode = True
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.gpu_ids = [gpu_id]
    setup_multi_processes(cfg)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

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

    return cfg, dataset, data_loader, model


def collect_points(dataset, data_loader, model, unknown_class_id, exclude_unknown_in_miou=False, tag='Set'):
    Xs, Ys = [], []

    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            pred = model(return_loss=False, rescale=True, **data)[0]
            pred = np.asarray(pred).astype(np.int64)

            gt = dataset.get_gt_seg_map_by_idx(idx)
            gt = np.asarray(gt).astype(np.int64)

            if pred.shape != gt.shape:
                print(f'[{tag}] Warning: shape mismatch at idx={idx}, pred={pred.shape}, gt={gt.shape}')
                continue

            unknown_ratio = compute_unknown_ratio_from_eval_gt_multi(gt, unknown_class_id)

            miou = compute_per_image_miou(
                pred,
                gt,
                ignore_index=255,
                exclude_unknown=exclude_unknown_in_miou,
                unknown_class_id=unknown_class_id
            )

            Xs.append(unknown_ratio)
            Ys.append(miou+10)

            file_name = get_sample_filename(dataset, idx)
            print(
                f'[{tag}] [{idx+1:04d}/{len(dataset)}] {file_name} | '
                f'unknown ratio: {unknown_ratio:.4f} | '
                f'mIoU: {miou*100:.2f}%'
            )

    return Xs, Ys


def bootstrap_linear_ci(X, Y, x_grid, n_boot=1000, ci=95, seed=42):
    """
    对线性拟合 y=ax+b 做 bootstrap，返回置信区间
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    preds = []

    if n < 2:
        return None, None

    for _ in range(n_boot):
        indices = rng.integers(0, n, n)
        Xb = X[indices]
        Yb = Y[indices]

        # 防止 bootstrap 后 X 完全重复导致 polyfit 不稳定
        if np.allclose(Xb, Xb[0]):
            continue

        try:
            coef = np.polyfit(Xb, Yb, deg=1)
            yb = coef[0] * x_grid + coef[1]
            preds.append(yb)
        except Exception:
            continue

    if len(preds) == 0:
        return None, None

    preds = np.array(preds)
    lower = np.percentile(preds, (100 - ci) / 2, axis=0)
    upper = np.percentile(preds, 100 - (100 - ci) / 2, axis=0)
    return lower, upper


def prepare_xy(X_list, Y_list, y_max=70.0):
    X = np.array(X_list, dtype=np.float64)
    Y = np.array(Y_list, dtype=np.float64)

    valid = ~(np.isnan(X) | np.isnan(Y) | np.isinf(X) | np.isinf(Y))
    X = X[valid]
    Y = Y[valid]

    if len(X) == 0:
        return None, None

    Y_percent = Y * 100.0

    keep = Y_percent <= y_max
    X = X[keep]
    Y_percent = Y_percent[keep]

    if len(X) == 0:
        return None, None

    sort_idx = np.argsort(X)
    X = X[sort_idx]
    Y_percent = Y_percent[sort_idx]

    return X, Y_percent


# ===================== 绘图函数 =====================
def plot_dual_linear_regression_with_ci(
        X1_list, Y1_list,
        X2_list, Y2_list,
        save_path='image_level_analysis_dual.png',
        dataset_name='Dataset',
        label1='Single Unknown',
        label2='Multiple Unknowns'):
    """
    在同一张图上绘制两组散点 + 两条线性拟合 + 两组置信区间
    """
    X1, Y1 = prepare_xy(X1_list, Y1_list, y_max=70.0)
    X2, Y2 = prepare_xy(X2_list, Y2_list, y_max=70.0)

    if X1 is None and X2 is None:
        print('两组数据都没有有效样本，无法绘图。')
        return

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'STIXGeneral'],
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False
    })

    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    x_line = np.linspace(0.0, 1.0, 300)

    # 配色
    color1 = '#4C72B0'   # 蓝
    line1 = '#2F5597'
    color2 = '#DD8452'   # 橙
    line2 = '#C44E52'

    # 第一组
    if X1 is not None and len(X1) >= 2:
        coef1 = np.polyfit(X1, Y1, deg=1)
        y_fit1 = coef1[0] * x_line + coef1[1]
        ci1_low, ci1_up = bootstrap_linear_ci(X1, Y1, x_line, n_boot=1000, ci=95, seed=42)

        ax.scatter(
            X1, Y1,
            s=28,
            c=color1,
            alpha=0.75,
            edgecolors='white',
            linewidths=0.6,
            marker='o',
            label=f'{label1} samples',
            zorder=3
        )
        ax.plot(
            x_line, y_fit1,
            color=line1,
            linewidth=2.0,
            linestyle='-',
            label=f'{label1} linear fit',
            zorder=4
        )
        if ci1_low is not None:
            ax.fill_between(
                x_line, ci1_low, ci1_up,
                color=color1,
                alpha=0.18,
                zorder=2
            )

        print(f'{label1} 线性拟合方程: y = {coef1[0]:.4f}x + {coef1[1]:.4f}')
        print(f'{label1} 有效样本数: {len(X1)}')

    # 第二组
    if X2 is not None and len(X2) >= 2:
        coef2 = np.polyfit(X2, Y2, deg=1)
        y_fit2 = coef2[0] * x_line + coef2[1]
        ci2_low, ci2_up = bootstrap_linear_ci(X2, Y2, x_line, n_boot=1000, ci=95, seed=123)

        ax.scatter(
            X2, Y2,
            s=28,
            c=color2,
            alpha=0.75,
            edgecolors='white',
            linewidths=0.6,
            marker='s',
            label=f'{label2} samples',
            zorder=3
        )
        ax.plot(
            x_line, y_fit2,
            color=line2,
            linewidth=2.0,
            linestyle='-',
            label=f'{label2} linear fit',
            zorder=4
        )
        if ci2_low is not None:
            ax.fill_between(
                x_line, ci2_low, ci2_up,
                color=color2,
                alpha=0.18,
                zorder=2
            )

        print(f'{label2} 线性拟合方程: y = {coef2[0]:.4f}x + {coef2[1]:.4f}')
        print(f'{label2} 有效样本数: {len(X2)}')

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 70.0)

    ax.set_xlabel('Unknown Class Ratio', fontsize=13)
    ax.set_ylabel('mIoU (%)', fontsize=13)

    ax.tick_params(axis='both', which='major', labelsize=11, direction='in', length=5, width=0.8)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.6)

    ax.grid(True, which='major', linestyle='--', linewidth=0.55, color='#D9D9D9', alpha=0.8)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('#333333')

    ax.legend(
        loc='best',
        fontsize=10,
        frameon=True,
        fancybox=False,
        edgecolor='#B0B0B0'
    )

    try:
        ax.set_box_aspect(1)
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'\n✅ 图像已保存: {save_path}')


# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='LoveDA', choices=['ISPRS', 'LoveDA'])

    # 第一组：当前这一组（单未知类）
    parser.add_argument('--config',
                        default='experiments/segformerb5/config_LoveDA/OSNet_40k_R2U.py',
                        help='config path for first set')
    parser.add_argument('--checkpoint',
                        default='/data/fywdata/fyw/UDA/OSUDA/MyNet/myresults_R2U_segformer/iter_40000.pth',
                        help='checkpoint path for first set')

    # 第二组：新增这一组（多未知类）
    parser.add_argument('--config2',
                        default='experiments/segformerb5/config_LoveDA/OSNet_40k_R2UV1.py',
                        help='config path for second set')
    parser.add_argument('--checkpoint2',
                        default='/data/fywdata/fyw/UDA/OSUDA/MyNet/myresults_R2U_segformerdel2/iter_40000.pth',
                        help='checkpoint path for second set')

    parser.add_argument('--gpu-id', type=int, default=4)
    parser.add_argument('--save-path', type=str, default='image_level_analysis_dual.png')
    parser.add_argument('--exclude-unknown-in-miou', action='store_true',
                        help='whether to exclude unknown class when computing per-image mIoU')

    parser.add_argument('--label1', type=str, default='Single Unknown')
    parser.add_argument('--label2', type=str, default='Multiple Unknowns')

    args = parser.parse_args()

    set_dataset_config(args.dataset)

    # 第一组
    print('\n' + '=' * 80)
    print('Collecting first set of points...')
    print('=' * 80)
    cfg1, dataset1, data_loader1, model1 = build_model_and_loader(
        args.config, args.checkpoint, args.gpu_id
    )
    X1, Y1 = collect_points(
        dataset1, data_loader1, model1,
        unknown_class_id=UNKNOWN_CLASS_ID,
        exclude_unknown_in_miou=args.exclude_unknown_in_miou,
        tag='Set1'
    )

    # 第二组
    print('\n' + '=' * 80)
    print('Collecting second set of points...')
    print('=' * 80)
    cfg2, dataset2, data_loader2, model2 = build_model_and_loader(
        args.config2, args.checkpoint2, args.gpu_id
    )
    X2, Y2 = collect_points(
        dataset2, data_loader2, model2,
        unknown_class_id=UNKNOWN_CLASS_ID,
        exclude_unknown_in_miou=args.exclude_unknown_in_miou,
        tag='Set2'
    )

    # 绘图
    plot_dual_linear_regression_with_ci(
        X1, Y1,
        X2, Y2,
        save_path=args.save_path,
        dataset_name=args.dataset,
        label1=args.label1,
        label2=args.label2
    )


if __name__ == '__main__':
    main()