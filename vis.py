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
from matplotlib import gridspec

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from mmseg.models import build_segmentor
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import setup_multi_processes


# ===================== 配置 =====================
UNKNOWN_CLASS_ID = -1


def set_dataset_config(dataset_name):
    global UNKNOWN_CLASS_ID
    if dataset_name == 'ISPRS':
        # 这里假设 ISPRS 的 unknown 在 eval label space 里是 5
        UNKNOWN_CLASS_ID = 5
    elif dataset_name == 'LoveDA':
        # 对于你的 LoveDA open-set 配置：
        # target classes = 7类，agricultural 是 unknown
        # eval label space 通过 get_gt_seg_map_by_idx() 映射成 0~6
        # agricultural -> 6
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

    参数:
        pred: HxW, ndarray, 预测类别索引
        gt:   HxW, ndarray, eval label space 下的GT
        ignore_index: 忽略标签
        exclude_unknown: 是否在mIoU中排除unknown类
        unknown_class_id: unknown类别id（当exclude_unknown=True时需要）

    返回:
        miou: float
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
    """尽量从dataset中获取当前样本文件名，仅用于打印"""
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
def plot_joint_regression_with_hist_and_residuals(
        X_list,
        Y_list,
        save_path='image_level_analysis.png',
        dataset_name='Dataset'):
    """
    绘制:
      1. 散点图 + 线性回归线
      2. X边缘直方图
      3. Y边缘直方图
      4. 残差箱式图
    """
    X = np.array(X_list, dtype=np.float64)
    Y = np.array(Y_list, dtype=np.float64)

    valid = ~(np.isnan(X) | np.isnan(Y) | np.isinf(X) | np.isinf(Y))
    X = X[valid]
    Y = Y[valid]

    if len(X) == 0:
        print('没有有效样本，无法绘图。')
        return

    # 回归
    lr = LinearRegression()
    lr.fit(X.reshape(-1, 1), Y)
    Y_fit = lr.predict(X.reshape(-1, 1))
    residuals = Y - Y_fit
    r2 = r2_score(Y, Y_fit)

    coef = float(lr.coef_[0])
    intercept = float(lr.intercept_)

    x_line = np.linspace(X.min(), X.max(), 200)
    y_line = lr.predict(x_line.reshape(-1, 1))

    # 布局
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(
        3, 2,
        width_ratios=[4.0, 1.2],
        height_ratios=[1.2, 4.0, 1.4],
        hspace=0.28,
        wspace=0.28
    )

    ax_histx = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_histy = fig.add_subplot(gs[1, 1])
    ax_box = fig.add_subplot(gs[2, 0])

    # ===== 主图：散点 + 回归线 =====
    ax_scatter.scatter(
        X, Y,
        s=45,
        alpha=0.75,
        color='#2E86AB',
        edgecolors='white',
        linewidths=0.5,
        label='Samples'
    )
    ax_scatter.plot(
        x_line, y_line,
        color='red',
        linewidth=2.0,
        label=f'Linear fit ($R^2={r2:.3f}$)'
    )

    ax_scatter.invert_xaxis()
    ax_scatter.set_xlabel('Unknown class ratio')
    ax_scatter.set_ylabel('Per-image mIoU')
    ax_scatter.set_title(
        f'{dataset_name}: Unknown Ratio vs Per-image mIoU\n'
        f'y = {coef:.4f}x + {intercept:.4f}'
    )
    ax_scatter.grid(alpha=0.3)
    ax_scatter.legend()

    # ===== 顶部直方图 =====
    ax_histx.hist(
        X,
        bins=20,
        color='#85C1E9',
        edgecolor='black',
        alpha=0.85
    )
    ax_histx.invert_xaxis()
    ax_histx.set_ylabel('Count')
    ax_histx.set_title('Histogram of Unknown Class Ratio')
    ax_histx.grid(alpha=0.2)
    ax_histx.tick_params(axis='x', labelbottom=False)

    # ===== 右侧直方图 =====
    ax_histy.hist(
        Y,
        bins=20,
        orientation='horizontal',
        color='#F5B7B1',
        edgecolor='black',
        alpha=0.85
    )
    ax_histy.set_xlabel('Count')
    ax_histy.set_title('Histogram of Per-image mIoU')
    ax_histy.grid(alpha=0.2)
    ax_histy.tick_params(axis='y', labelleft=False)

    # ===== 残差箱式图 =====
    ax_box.boxplot(
        residuals,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor='#A9DFBF', color='black'),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(
            marker='o',
            markerfacecolor='orange',
            markeredgecolor='black',
            markersize=5,
            linestyle='none'
        )
    )
    ax_box.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax_box.set_xlabel('Residuals (Observed mIoU - Predicted mIoU)')
    ax_box.set_title('Residual Boxplot')
    ax_box.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'\n✅ 图像已保存: {save_path}')
    print(f'有效样本数: {len(X)}')
    print(f'线性回归方程: y = {coef:.6f}x + {intercept:.6f}')
    print(f'R² = {r2:.6f}')


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

    # 构建dataset / dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    # 构建模型
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
            # 官方推理
            pred = model(return_loss=False, rescale=True, **data)[0]
            pred = np.asarray(pred).astype(np.int64)

            # 正确获取与当前样本严格对齐的 eval GT
            gt = dataset.get_gt_seg_map_by_idx(idx)
            gt = np.asarray(gt).astype(np.int64)

            # 检查尺寸
            if pred.shape != gt.shape:
                print(f'Warning: shape mismatch at idx={idx}, pred={pred.shape}, gt={gt.shape}')
                # 一般 rescale=True 后应一致；如不一致可按需插值/跳过
                continue

            # 基于 eval GT 统计 unknown ratio
            unknown_ratio = compute_unknown_ratio_from_eval_gt(gt, UNKNOWN_CLASS_ID)

            # 计算单图mIoU
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
                f'per-image mIoU: {miou:.4f}'
            )

    plot_joint_regression_with_hist_and_residuals(
        Xs,
        Ys,
        save_path=args.save_path,
        dataset_name=args.dataset
    )


if __name__ == '__main__':
    main()