import os
import torch
import numpy as np
import mmcv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from torchvision import transforms
from PIL import Image

# MMSegmentation 核心导入
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint
from mmseg.utils import setup_multi_processes

# ===================== 1. 配置参数（直接适配你的代码） =====================
# 你的配置文件路径
CONFIG_PATH = "experiments/segformerb5/config/OSNet_40k_Potsdam2Vaihingen.py"
# 你的模型权重路径
CHECKPOINT_PATH = "/data/fywdata/fyw/UDA/OSUDA/MyNet/myresults_P2V_segformer/iter_8000.pth"
# 可视化保存路径
SAVE_DIR = "/data/fywdata/fyw/UDA/OSUDA/MyNet/feature_visualization/"
# GPU设置
DEVICE = "cuda:3"
# 采样数量（t-SNE用）
TSNE_SAMPLE_PER_CLASS = 150

# 类别定义（严格对齐你的数据集）
KNOWN_CLASSES = ['impervious_surface', 'building', 'low_vegetation', 'tree', 'car']
UNKNOWN_CLASS = ['clutter']
ALL_CLASSES = KNOWN_CLASSES + UNKNOWN_CLASS
# 调色板（对齐你的数据集）
PALETTE = np.array([
    [255, 255, 255],  # 0: 不透水面
    [0, 0, 255],  # 1: 建筑
    [0, 255, 255],  # 2: 低植被
    [0, 255, 0],  # 3: 树
    [255, 255, 0],  # 4: 车
    [255, 0, 0],  # 5: 杂波(未知类)
]) / 255.0

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)


# ===================== 2. 加载模型和数据集 =====================
def build_model_and_dataset():
    # 加载配置
    cfg = mmcv.Config.fromfile(CONFIG_PATH)
    setup_multi_processes(cfg)

    # 关闭训练配置，设置测试模式
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    cfg.gpu_ids = [3]

    # 构建源域数据集（Potsdam）和目标域数据集（Vaihingen）
    source_dataset = build_dataset(cfg.data.train)  # 源域
    target_dataset = build_dataset(cfg.data.test)  # 目标域

    # 构建模型
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    # 加载权重
    load_checkpoint(model, CHECKPOINT_PATH, map_location=DEVICE)
    model.to(DEVICE)
    model.eval()  # 推理模式

    # 构建数据加载器
    source_loader = build_dataloader(source_dataset, samples_per_gpu=1, workers_per_gpu=2, shuffle=False, dist=False)
    target_loader = build_dataloader(target_dataset, samples_per_gpu=1, workers_per_gpu=2, shuffle=False, dist=False)

    return model, source_loader, target_loader


# ===================== 3. 提取特征函数 =====================
@torch.no_grad()
def extract_features(model, data_loader, is_source=True):
    """提取特征+标签：返回 特征向量, 类别标签, 域标签(0=源域,1=目标域)"""
    features_list = []
    labels_list = []
    domain_list = []

    for data in data_loader:
        # 数据加载
        img = data['img'][0].to(DEVICE)
        gt = data['gt_semantic_seg'][0].cpu().numpy().squeeze()

        # 前向传播，提取主干网络最后一层特征（核心对齐特征）
        # 适配你的SegFormer主干
        backbone_feat = model.backbone_s(img)[-1]  # [1, 512, H/32, W/32]
        # 全局平均池化得到向量特征
        feat_vector = torch.nn.functional.adaptive_avg_pool2d(backbone_feat, 1).squeeze().cpu().numpy()

        # 获取有效标签（忽略255）
        valid_mask = (gt != 255)
        if not np.any(valid_mask):
            continue
        gt_valid = gt[valid_mask]

        # 采样一个主类别
        main_label = np.bincount(gt_valid).argmax()
        if main_label >= len(ALL_CLASSES):
            continue

        # 保存
        features_list.append(feat_vector)
        labels_list.append(main_label)
        domain_list.append(0 if is_source else 1)

        # 控制采样数量
        if len(features_list) >= TSNE_SAMPLE_PER_CLASS * 6:
            break

    return np.array(features_list), np.array(labels_list), np.array(domain_list)


# ===================== 4. t-SNE 特征可视化（核心：对齐+区分） =====================
def visualize_tsne(source_feat, source_label, source_domain,
                   target_feat, target_label, target_domain):
    # 合并数据
    feat_all = np.concatenate([source_feat, target_feat], axis=0)
    label_all = np.concatenate([source_label, target_label], axis=0)
    domain_all = np.concatenate([source_domain, target_domain], axis=0)

    # t-SNE降维
    print("正在计算t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    feat_2d = tsne.fit_transform(feat_all)

    # 绘图：分两种模式（1. 按类别 2. 按域）
    plt.figure(figsize=(16, 8))

    # 子图1：按类别着色（验证未知类区分）
    plt.subplot(1, 2, 1)
    for i, cls_name in enumerate(ALL_CLASSES):
        mask = (label_all == i)
        if not np.any(mask):
            continue
        # 未知类特殊标记
        marker = '*' if cls_name in UNKNOWN_CLASS else 'o'
        size = 120 if cls_name in UNKNOWN_CLASS else 80
        plt.scatter(feat_2d[mask, 0], feat_2d[mask, 1],
                    c=[PALETTE[i]], label=cls_name, marker=marker, s=size, alpha=0.8)
    plt.title("Feature Clustering (Known vs Unknown Classes)", fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.axis('off')

    # 子图2：按域着色（验证域对齐）
    plt.subplot(1, 2, 2)
    # 源域
    mask_s = (domain_all == 0)
    plt.scatter(feat_2d[mask_s, 0], feat_2d[mask_s, 1],
                c='blue', label='Source Domain', alpha=0.6, s=60)
    # 目标域已知类
    mask_t_known = (domain_all == 1) & (label_all < len(KNOWN_CLASSES))
    plt.scatter(feat_2d[mask_t_known, 0], feat_2d[mask_t_known, 1],
                c='green', label='Target Known Classes', alpha=0.6, s=60)
    # 目标域未知类
    mask_t_unknown = (domain_all == 1) & (label_all == len(KNOWN_CLASSES))
    plt.scatter(feat_2d[mask_t_unknown, 0], feat_2d[mask_t_unknown, 1],
                c='red', label='Target Unknown Class (Clutter)', marker='*', s=120)

    plt.title("Domain Alignment (Source vs Target)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "tsne_feature_alignment.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("t-SNE可视化已保存：tsne_feature_alignment.png")


# ===================== 5. 特征热力图可视化 =====================
@torch.no_grad()
def visualize_feature_heatmap(model, target_loader):
    """可视化目标域特征热力图：已知类+未知类的特征激活"""
    data_iter = iter(target_loader)
    # 取3张图：已知类2张 + 未知类1张
    for _ in range(5):
        data = next(data_iter)
    img = data['img'][0].to(DEVICE)
    gt = data['gt_semantic_seg'][0].cpu().numpy().squeeze()

    # 提取特征
    feat = model.backbone_s(img)[-1]
    # 均值热力图
    heatmap = torch.mean(feat, dim=1).squeeze().cpu().numpy()
    # 归一化
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # 绘图
    plt.figure(figsize=(15, 5))

    # 原图
    img_np = img.squeeze().cpu().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("Original Image", fontweight='bold')
    plt.axis('off')

    # 标签图
    plt.subplot(1, 3, 2)
    gt[gt == 255] = 0
    plt.imshow(gt, cmap=ListedColormap(PALETTE))
    plt.title("Ground Truth (Red=Unknown)", fontweight='bold')
    plt.axis('off')

    # 特征热力图
    plt.subplot(1, 3, 3)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Feature Heatmap", fontweight='bold')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "feature_heatmap.png"), dpi=300)
    plt.close()
    print("特征热力图已保存：feature_heatmap.png")


# ===================== 6. 主函数 =====================
if __name__ == "__main__":
    # 1. 加载模型和数据
    model, source_loader, target_loader = build_model_and_dataset()
    print("模型与数据集加载完成！")

    # 2. 提取源域+目标域特征
    print("正在提取源域特征...")
    s_feat, s_label, s_domain = extract_features(model, source_loader, is_source=True)
    print("正在提取目标域特征...")
    t_feat, t_label, t_domain = extract_features(model, target_loader, is_source=False)

    # 3. 核心t-SNE可视化（对齐+区分）
    visualize_tsne(s_feat, s_label, s_domain, t_feat, t_label, t_domain)

    # 4. 特征热力图可视化
    visualize_feature_heatmap(model, target_loader)

    print("\n✅ 所有特征可视化完成！保存路径：", SAVE_DIR)