# tsne.py
import argparse
import os
import os.path as osp
import numpy as np
import torch
import mmcv
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.cnn.utils import revert_sync_batchnorm
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from PIL import Image
import cv2

from mmseg.models import build_segmentor
from mmseg.utils import setup_multi_processes

# ===================== 全局配置 =====================
SOURCE_INCLUDED_CLASSES = [
    'impervious_surface', 'building', 'low_vegetation', 'tree', 'car'
]
TARGET_INCLUDED_CLASSES = [
    'impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter'
]

CLASS_COLORS = {
    'impervious_surface': '#1f77b4',
    'building':           '#ff7f0e',
    'low_vegetation':     '#2ca02c',
    'tree':               '#d62728',
    'car':                '#17becf',
    'clutter':            '#8B00FF',
}
UNKNOWN_COLOR = '#8B00FF'

# ImageNet 归一化参数（与训练时保持一致）
IMG_MEAN = np.array([123.675, 116.28,  103.53],  dtype=np.float32)
IMG_STD  = np.array([58.395,  57.12,   57.375],  dtype=np.float32)
# ====================================================


def parse_args():
    parser = argparse.ArgumentParser(description='t-SNE feature visualization')
    parser.add_argument('--config',
                        default='experiments/segformerb5/config/OSNet_40k_Potsdam2Vaihingen.py')
    parser.add_argument('--checkpoint',
                        default='/data/fywdata/fyw/UDA/OSUDA/MyNet/myresults_P2V_segformer/iter_8000.pth')

    # 直接指定源域和目标域的图像/标签目录（不走训练数据集类）
    parser.add_argument('--src-img-dir',
                        default='/data/fywdata/ISPRS/Potsdam_IRRG/img_dir/train',
                        help='源域图像目录，如 data/Potsdam/img_dir/val')
    parser.add_argument('--src-ann-dir',
                        default='/data/fywdata/ISPRS/Potsdam_IRRG/ann_dir/train',
                        help='源域标签目录，如 data/Potsdam/ann_dir/val')
    parser.add_argument('--tgt-img-dir',
                        default='/data/fywdata/ISPRS/Vaihingen_IRRG/img_dir/val',
                        help='目标域图像目录，如 data/Vaihingen/img_dir/test')
    parser.add_argument('--tgt-ann-dir',
                        default='/data/fywdata/ISPRS/Vaihingen_IRRG/ann_dir/val',
                        help='目标域标签目录，如 data/Vaihingen/ann_dir/test')

    parser.add_argument('--img-suffix',  default='.png')
    parser.add_argument('--ann-suffix',  default='.png')
    parser.add_argument('--img-size',    type=int, nargs=2, default=[512, 512],
                        help='输入模型的图像大小 H W')
    parser.add_argument('--save-path',   default='tsne_visualization.png')
    parser.add_argument('--gpu-id',      type=int, default=0)
    parser.add_argument('--max-pixels',  type=int, default=500,
                        help='每类每张图最多采样像素数')
    parser.add_argument('--num-images',  type=int, default=20,
                        help='每个域最多处理多少张图')
    parser.add_argument('--perplexity',  type=float, default=40.0)
    parser.add_argument('--tsne-iter',   type=int, default=1000)
    return parser.parse_args()


# ===================== Hook =====================

class BackboneHook:
    def __init__(self):
        self.outputs = None
        self._handle = None

    def register(self, module):
        self._handle = module.register_forward_hook(self._fn)

    def _fn(self, module, inp, output):
        if isinstance(output, (tuple, list)):
            self.outputs = [o.detach().cpu() for o in output]
        else:
            self.outputs = [output.detach().cpu()]

    def get_last(self):
        """取最后一个 stage 输出 (B, C, H, W)"""
        return None if self.outputs is None else self.outputs[-1]

    def remove(self):
        if self._handle is not None:
            self._handle.remove()


# ===================== 数据读取工具 =====================

def collect_file_pairs(img_dir, ann_dir, img_suffix, ann_suffix):
    """
    收集 (img_path, ann_path) 对，自动匹配文件名（去掉后缀后名字相同）。
    """
    img_files = sorted([
        f for f in os.listdir(img_dir) if f.endswith(img_suffix)
    ])
    pairs = []
    for fname in img_files:
        stem    = fname[: -len(img_suffix)]
        ann_f   = stem + ann_suffix
        ann_path = osp.join(ann_dir, ann_f)
        if osp.exists(ann_path):
            pairs.append((osp.join(img_dir, fname), ann_path))
        else:
            print(f'  [WARN] ann not found: {ann_path}, skip')
    print(f'  Found {len(pairs)} valid image-annotation pairs in {img_dir}')
    return pairs


def load_and_preprocess_img(img_path, target_hw):
    """
    读图 → BGR→RGB → resize → 归一化 → (1,3,H,W) float32 Tensor
    """
    img = cv2.imread(img_path)           # BGR
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {img_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)  # RGB float
    img = cv2.resize(img, (target_hw[1], target_hw[0]),
                     interpolation=cv2.INTER_LINEAR)
    img = (img - IMG_MEAN) / IMG_STD
    img = img.transpose(2, 0, 1)        # (3,H,W)
    return torch.from_numpy(img).unsqueeze(0)   # (1,3,H,W)


def load_label(ann_path, target_hw):
    """
    读标签 → 最近邻 resize → (H,W) int32 numpy
    """
    lbl = np.array(Image.open(ann_path))
    if lbl.ndim == 3:                   # RGBA / RGB label-color map 情况
        lbl = lbl[:, :, 0]             # 取第一通道，视数据集而定
    lbl = lbl.astype(np.int32)
    lbl_img = Image.fromarray(lbl.astype(np.uint8))
    lbl_img = lbl_img.resize((target_hw[1], target_hw[0]), Image.NEAREST)
    return np.array(lbl_img, dtype=np.int32)


# ===================== 特征提取 =====================

def extract_features(model, backbone, file_pairs,
                     num_images, max_pixels_per_class,
                     num_classes, img_size, device, domain_name=''):
    """
    Returns: feats (N,C), labels (N,), domain (N,)
    """
    hook = BackboneHook()
    hook.register(backbone)

    buckets = {c: [] for c in range(num_classes)}
    is_source = (domain_name == 'source')

    used = min(num_images, len(file_pairs))
    file_pairs_used = file_pairs[:used]

    for i, (img_path, ann_path) in enumerate(file_pairs_used):
        print(f'  [{domain_name}] {i+1}/{used}  {osp.basename(img_path)}', end='\r')

        # ---------- 读取并预处理 ----------
        try:
            img_tensor = load_and_preprocess_img(img_path, img_size).to(device)
            gt_full    = load_label(ann_path, img_size)   # (H,W)
        except Exception as e:
            print(f'\n  [WARN] skip {img_path}: {e}')
            continue

        # ---------- 前向 ----------
        hook.outputs = None
        with torch.no_grad():
            backbone(img_tensor)

        feat = hook.get_last()          # (1, C, fH, fW)
        if feat is None:
            print(f'\n  [WARN] hook got nothing for {img_path}')
            continue

        _, C, fH, fW = feat.shape
        feat_np = feat[0].permute(1, 2, 0).reshape(-1, C).numpy()  # (fH*fW, C)

        # ---------- 将 gt 缩放到 feature 分辨率 ----------
        gt_small = np.array(
            Image.fromarray(gt_full.astype(np.uint8)).resize(
                (fW, fH), Image.NEAREST),
            dtype=np.int32
        ).reshape(-1)   # (fH*fW,)

        # ---------- 按类采样 ----------
        for lbl in range(num_classes):
            idx = np.where(gt_small == lbl)[0]
            if len(idx) == 0:
                continue
            if len(idx) > max_pixels_per_class:
                idx = np.random.choice(idx, max_pixels_per_class, replace=False)
            buckets[lbl].append(feat_np[idx])

    hook.remove()
    print()

    # ---------- 整合 ----------
    all_feats, all_labels = [], []
    cap = max_pixels_per_class * used
    for lbl, lst in buckets.items():
        if len(lst) == 0:
            print(f'  [WARN] [{domain_name}] class {lbl} has 0 pixels')
            continue
        f = np.concatenate(lst, axis=0)
        if len(f) > cap:
            idx = np.random.choice(len(f), cap, replace=False)
            f = f[idx]
        all_feats.append(f)
        all_labels.append(np.full(len(f), lbl, dtype=np.int32))
        print(f'  [{domain_name}] class {lbl}: {len(f)} pixels collected')

    if not all_feats:
        return (np.zeros((0, 1), dtype=np.float32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32))

    all_feats  = np.concatenate(all_feats,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_domain = np.full(len(all_feats), 0 if is_source else 1, dtype=np.int32)
    return all_feats, all_labels, all_domain


# ===================== 绘图 =====================

def plot_tsne(tsne_xy, all_labels, all_domain,
              source_classes, target_classes, save_path, title='t-SNE'):

    fig, ax = plt.subplots(figsize=(12, 10))

    tgt_colors = [CLASS_COLORS.get(c, UNKNOWN_COLOR) for c in target_classes]
    src_colors = [CLASS_COLORS.get(c, UNKNOWN_COLOR) for c in source_classes]

    for dom in [1, 0]:   # 目标域先画（底层），源域后画（顶层）
        for lbl in np.unique(all_labels):
            mask = (all_domain == dom) & (all_labels == lbl)
            if mask.sum() == 0:
                continue
            pts = tsne_xy[mask]

            if dom == 0:    # 源域
                colors = src_colors
                marker, s, alpha, zorder = '^', 25, 0.80, 3
            else:           # 目标域
                colors = tgt_colors
                marker, s, alpha, zorder = 'o', 15, 0.45, 2

            color = colors[lbl] if lbl < len(colors) else UNKNOWN_COLOR
            ax.scatter(pts[:, 0], pts[:, 1],
                       c=color, marker=marker,
                       s=s, alpha=alpha,
                       linewidths=0, zorder=zorder)

    # legend
    legend_elems = []
    for cls in target_classes:
        color = CLASS_COLORS.get(cls, UNKNOWN_COLOR)
        label = cls if cls in source_classes else f'{cls}  [unknown]'
        legend_elems.append(mpatches.Patch(color=color, label=label))
    legend_elems += [
        Line2D([0], [0], marker='^', color='gray', linestyle='None',
               markersize=9, label='Source  (△)'),
        Line2D([0], [0], marker='o', color='gray', linestyle='None',
               markersize=9, label='Target  (●)'),
    ]
    ax.legend(handles=legend_elems, loc='upper right',
              fontsize=9, framealpha=0.85)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)

    plt.tight_layout()
    os.makedirs(osp.dirname(osp.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f't-SNE figure saved → {save_path}')
    plt.close()


# ===================== 从 config 自动解析目录 =====================

def auto_find_dirs(cfg):
    """
    尝试从 cfg.data.val / cfg.data.test 中解析图像和标签目录。
    支持 UDA 配置中常见的几种字段名。
    """
    def _get(ds_cfg, *keys):
        for k in keys:
            v = ds_cfg.get(k, None)
            if v is not None:
                return v
        return None

    src_img, src_ann, tgt_img, tgt_ann = None, None, None, None

    # 源域: cfg.data.val
    val_cfg = cfg.data.get('val', None)
    if val_cfg is not None:
        data_root = _get(val_cfg, 'data_root', 'root')
        img_dir   = _get(val_cfg, 'img_dir')
        ann_dir   = _get(val_cfg, 'ann_dir')
        if data_root and img_dir:
            src_img = osp.join(data_root, img_dir)
        if data_root and ann_dir:
            src_ann = osp.join(data_root, ann_dir)

    # 目标域: cfg.data.test
    test_cfg = cfg.data.get('test', None)
    if test_cfg is not None:
        data_root = _get(test_cfg, 'data_root', 'root')
        img_dir   = _get(test_cfg, 'img_dir')
        ann_dir   = _get(test_cfg, 'ann_dir')
        if data_root and img_dir:
            tgt_img = osp.join(data_root, img_dir)
        if data_root and ann_dir:
            tgt_ann = osp.join(data_root, ann_dir)

    return src_img, src_ann, tgt_img, tgt_ann


# ===================== main =====================

def main():
    args = parse_args()
    cfg  = mmcv.Config.fromfile(args.config)
    setup_multi_processes(cfg)

    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    # ---- 构建模型 ----
    cfg.model.pretrained = None
    cfg.model.train_cfg  = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = revert_sync_batchnorm(model)
    model = MMDataParallel(model, device_ids=[args.gpu_id])
    model.to(device)
    model.eval()
    print('Model loaded.')

    # 取 backbone_s（真实模型在 model.module）
    backbone = model.module.backbone_s

    # ---- 确定数据目录 ----
    src_img_dir = args.src_img_dir
    src_ann_dir = args.src_ann_dir
    tgt_img_dir = args.tgt_img_dir
    tgt_ann_dir = args.tgt_ann_dir

    # 若命令行未指定，尝试从 config 自动解析
    if not all([src_img_dir, src_ann_dir, tgt_img_dir, tgt_ann_dir]):
        print('Auto-detecting data dirs from config...')
        a, b, c, d = auto_find_dirs(cfg)
        src_img_dir = src_img_dir or a
        src_ann_dir = src_ann_dir or b
        tgt_img_dir = tgt_img_dir or c
        tgt_ann_dir = tgt_ann_dir or d

    print(f'  src_img: {src_img_dir}')
    print(f'  src_ann: {src_ann_dir}')
    print(f'  tgt_img: {tgt_img_dir}')
    print(f'  tgt_ann: {tgt_ann_dir}')

    for d, name in [(src_img_dir, 'src_img'), (src_ann_dir, 'src_ann'),
                    (tgt_img_dir, 'tgt_img'), (tgt_ann_dir, 'tgt_ann')]:
        if d is None or not osp.isdir(d):
            raise FileNotFoundError(
                f'{name} 目录不存在或未指定: {d}\n'
                f'请通过命令行参数 --src-img-dir / --src-ann-dir / '
                f'--tgt-img-dir / --tgt-ann-dir 手动指定')

    img_size = tuple(args.img_size)    # (H, W)

    # ---- 收集文件对 ----
    print('Collecting source file pairs...')
    src_pairs = collect_file_pairs(src_img_dir, src_ann_dir,
                                   args.img_suffix, args.ann_suffix)
    print('Collecting target file pairs...')
    tgt_pairs = collect_file_pairs(tgt_img_dir, tgt_ann_dir,
                                   args.img_suffix, args.ann_suffix)

    if not src_pairs:
        raise RuntimeError(f'源域无有效图像对: {src_img_dir}')
    if not tgt_pairs:
        raise RuntimeError(f'目标域无有效图像对: {tgt_img_dir}')

    # ---- 提取特征 ----
    print(f'\n[Source] extracting (max {args.num_images} images)...')
    src_feats, src_labels, src_domain = extract_features(
        model, backbone, src_pairs,
        num_images=args.num_images,
        max_pixels_per_class=args.max_pixels,
        num_classes=len(SOURCE_INCLUDED_CLASSES),
        img_size=img_size, device=device, domain_name='source')

    print(f'\n[Target] extracting (max {args.num_images} images)...')
    tgt_feats, tgt_labels, tgt_domain = extract_features(
        model, backbone, tgt_pairs,
        num_images=args.num_images,
        max_pixels_per_class=args.max_pixels,
        num_classes=len(TARGET_INCLUDED_CLASSES),
        img_size=img_size, device=device, domain_name='target')

    print(f'\nSource: {src_feats.shape}  Target: {tgt_feats.shape}')

    if src_feats.shape[0] == 0 or tgt_feats.shape[0] == 0:
        raise RuntimeError('特征提取结果为空，请检查数据目录和标签值范围。')

    all_feats  = np.concatenate([src_feats,  tgt_feats],  axis=0)
    all_labels = np.concatenate([src_labels, tgt_labels], axis=0)
    all_domain = np.concatenate([src_domain, tgt_domain], axis=0)
    print(f'Total: {len(all_feats)} samples, dim={all_feats.shape[1]}')

    # ---- t-SNE ----
    perp = min(args.perplexity, len(all_feats) - 1)
    print(f'Running t-SNE (perplexity={perp}, max_iter={args.tsne_iter})...')
    tsne    = TSNE(n_components=2, perplexity=perp,
                   n_iter=args.tsne_iter,
                   random_state=42, verbose=1)
    tsne_xy = tsne.fit_transform(all_feats)
    print('t-SNE done.')

    # ---- 绘图 ----
    plot_tsne(tsne_xy, all_labels, all_domain,
              source_classes=SOURCE_INCLUDED_CLASSES,
              target_classes=TARGET_INCLUDED_CLASSES,
              save_path=args.save_path,
              title='t-SNE Visualization  |  P→V  |  △ Source  ● Target')


if __name__ == '__main__':
    main()