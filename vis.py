import argparse
import os
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from PIL import Image

from mmseg.models import build_segmentor
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import setup_multi_processes

# ===================== 全局配置（和你tsne一致） =====================
UNKNOWN_CLASS_ID = -1
TARGET_CLASS_NUM = -1

def set_dataset_config(dataset_name):
    global UNKNOWN_CLASS_ID, TARGET_CLASS_NUM
    if dataset_name == 'ISPRS':
        UNKNOWN_CLASS_ID = 5
        TARGET_CLASS_NUM = 6
    elif dataset_name == 'LoveDA':
        UNKNOWN_CLASS_ID = 6
        TARGET_CLASS_NUM = 7

# ===================== 单图 mIoU 计算 =====================
def compute_per_image_miou(pred_mask, gt_mask, num_classes, ignore_id):
    ious = []
    for cls in range(num_classes):
        if cls == ignore_id:
            continue
        intersection = np.logical_and(pred_mask == cls, gt_mask == cls).sum()
        union = np.logical_or(pred_mask == cls, gt_mask == cls).sum()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0

# ===================== 未知类别占比 =====================
def compute_unknown_ratio(gt_mask):
    return np.sum(gt_mask == UNKNOWN_CLASS_ID) / gt_mask.size

# ===================== 绘图（你的要求：X降序 1→0） =====================
def plot_image_level_fit(X_list, Y_list, save_path='image_level_fit.png'):
    X = np.array(X_list)
    Y = np.array(Y_list)
    sorted_idx = np.argsort(X)[::-1]
    X, Y = X[sorted_idx], Y[sorted_idx]

    lr = LinearRegression()
    lr.fit(X.reshape(-1, 1), Y)
    r2 = r2_score(Y, lr.predict(X.reshape(-1, 1)))

    plt.figure(figsize=(7, 6))
    plt.scatter(X, Y, s=60, alpha=0.7, color='#2E86AB')
    plt.plot(X, lr.predict(X.reshape(-1,1)), 'r-', linewidth=2, label=f'$R^2$={r2:.3f}')
    plt.gca().invert_xaxis()
    plt.xlabel('Unknown class ratio', fontsize=12)
    plt.ylabel('Per-image mIoU', fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n✅ 图像已保存: {save_path}")

# ===================== 主函数（完全复用你的官方测试逻辑） =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='LoveDA', choices=['ISPRS', 'LoveDA'])
    parser.add_argument('--config', default='experiments/segformerb5/config_LoveDA/OSNet_40k_R2U.py', help='test config path')
    parser.add_argument('--checkpoint', default='/data/fywdata/fyw/UDA/OSUDA/MyNet/myresults_R2U_segformer/iter_4000.pth', help='checkpoint path')
    parser.add_argument('--gpu-id', type=int, default=4)
    args = parser.parse_args()

    set_dataset_config(args.dataset)
    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    cfg.gpu_ids = [args.gpu_id]
    setup_multi_processes(cfg)

    # --------------- 完全和你测试代码一样 ---------------
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=4, dist=False, shuffle=False)

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = revert_sync_batchnorm(model)
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    model.eval()

    X_list, Y_list = [], []
    print("\n开始推理并计算指标...")

    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            # 单图推理（官方接口，零报错）
            pred_logit = model(return_loss=False, rescale=True, **data)
            pred_mask = pred_logit[0].argmax(axis=0)

            # 获取原图GT
            gt_mask = data['gt_semantic_seg'][0].squeeze().cpu().numpy()
            gt_mask = gt_mask.astype(np.uint8)

            # 计算横纵坐标
            x = compute_unknown_ratio(gt_mask)
            y = compute_per_image_miou(pred_mask, gt_mask, TARGET_CLASS_NUM, UNKNOWN_CLASS_ID)

            X_list.append(x)
            Y_list.append(y)
            print(f"[{idx+1}] 未知占比: {x:.3f} | mIoU: {y:.3f}")

    plot_image_level_fit(X_list, Y_list, f'image_level_{args.dataset}.png')

if __name__ == '__main__':
    main()