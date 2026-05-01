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
from PIL import Image
import cv2

from mmseg.models import build_segmentor
from mmseg.utils import setup_multi_processes

# ===================== 全局配置 =====================
CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter')
PALETTE = [
    [0,   0,   0  ],   # 0: impervious_surface  黑
    [0,   0,   255],   # 1: building            蓝
    [0,   255, 255],   # 2: low_vegetation      青
    [0,   255, 0  ],   # 3: tree                绿
    [255, 255, 0  ],   # 4: car                 黄
    [255, 0,   0  ],   # 5: clutter             红
]

# 将 PALETTE 归一化为 [0,1]，供 matplotlib 使用
COLORS_NORM = [[r/255., g/255., b/255.] for r, g, b in PALETTE]

SOURCE_CLASS_NUM = 5   # 源域有 5 类（无 clutter）
TARGET_CLASS_NUM = 6   # 目标域有 6 类（含 clutter）

# ImageNet 归一化参数
IMG_MEAN = np.array([123.675, 116.28,  103.53],  dtype=np.float32)
IMG_STD  = np.array([58.395,  57.12,   57.375],  dtype=np.float32)
# ====================================================


def parse_args():
    parser = argparse.ArgumentParser(description='t-SNE feature visualization')
    parser.add_argument('--config',
                        default='experiments/deeplabv3/config/OSNet_40k_Potsdam2Vaihingen.py')
    parser.add_argument('--checkpoint',
                        default='/data/fywdata/fyw/UDA/OSUDA/MyNet/myresults_P2V_deeplab/iter_8000.pth')

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
    parser.add_argument('--img-size',    type=int, nargs=2, default=[512, 512])
    parser.add_argument('--save-path',   default='tsne_visualization.png')
    parser.add_argument('--gpu-id',      type=int, default=0)
    parser.add_argument('--max-pixels',  type=int, default=300,
                        help='每类每张图最多采样像素数')
    parser.add_argument('--num-images',  type=int, default=20)
    parser.add_argument('--perplexity',  type=float, default=50.0)
    parser.add_argument('--tsne-iter',   type=int, default=2000)
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
        return None if self.outputs is None else self.outputs[-1]

    def remove(self):
        if self._handle is not None:
            self._handle.remove()


# ===================== 数据读取工具 =====================

def collect_file_pairs(img_dir, ann_dir, img_suffix, ann_suffix):
    img_files = sorted([
        f for f in os.listdir(img_dir) if f.endswith(img_suffix)
    ])
    pairs = []
    for fname in img_files:
        stem     = fname[: -len(img_suffix)]
        ann_path = osp.join(ann_dir, stem + ann_suffix)
        if osp.exists(ann_path):
            pairs.append((osp.join(img_dir, fname), ann_path))
        else:
            print(f'  [WARN] ann not found: {ann_path}, skip')
    print(f'  Found {len(pairs)} valid pairs in {img_dir}')
    return pairs


def load_and_preprocess_img(img_path, target_hw):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Cannot read: {img_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = cv2.resize(img, (target_hw[1], target_hw[0]),
                     interpolation=cv2.INTER_LINEAR)
    img = (img - IMG_MEAN) / IMG_STD
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0)


def load_label(ann_path, target_hw):
    lbl = np.array(Image.open(ann_path))
    if lbl.ndim == 3:
        lbl = lbl[:, :, 0]
    lbl = lbl.astype(np.int32)
    lbl_img = Image.fromarray(lbl.astype(np.uint8))
    lbl_img = lbl_img.resize((target_hw[1], target_hw[0]), Image.NEAREST)
    return np.array(lbl_img, dtype=np.int32)


# ===================== 特征提取 =====================

def extract_features(model, backbone, file_pairs,
                     num_images, max_pixels_per_class,
                     num_classes, img_size, device, domain_name=''):
    hook = BackboneHook()
    hook.register(backbone)

    # ---- 用 dict 存每个类的特征列表 ----
    buckets = {c: [] for c in range(num_classes)}
    used    = min(num_images, len(file_pairs))

    for i, (img_path, ann_path) in enumerate(file_pairs[:used]):
        print(f'  [{domain_name}] {i+1}/{used}  {osp.basename(img_path)}', end='\r')
        try:
            img_tensor = load_and_preprocess_img(img_path, img_size).to(device)
            gt_full    = load_label(ann_path, img_size)
        except Exception as e:
            print(f'\n  [WARN] skip {img_path}: {e}')
            continue

        hook.outputs = None
        with torch.no_grad():
            backbone(img_tensor)

        feat = hook.get_last()
        if feat is None:
            continue

        _, C, fH, fW = feat.shape
        feat_np = feat[0].permute(1, 2, 0).reshape(-1, C).numpy()  # (fH*fW, C)

        gt_small = np.array(
            Image.fromarray(gt_full.astype(np.uint8)).resize(
                (fW, fH), Image.NEAREST),
            dtype=np.int32
        ).reshape(-1)

        # ---- 忽略 255 (unlabeled) ----
        for lbl in range(num_classes):
            idx = np.where(gt_small == lbl)[0]
            if len(idx) == 0:
                continue
            if len(idx) > max_pixels_per_class:
                idx = np.random.choice(idx, max_pixels_per_class, replace=False)
            buckets[lbl].append(feat_np[idx])

    hook.remove()
    print()

    # ---- 整合，对总量做上限控制 ----
    cap = max_pixels_per_class * used
    all_feats, all_labels = [], []
    for lbl, lst in buckets.items():
        if not lst:
            print(f'  [WARN] [{domain_name}] class {lbl} ({CLASSES[lbl]}) has 0 pixels')
            continue
        f = np.concatenate(lst, axis=0)
        if len(f) > cap:
            idx = np.random.choice(len(f), cap, replace=False)
            f   = f[idx]
        all_feats.append(f)
        all_labels.append(np.full(len(f), lbl, dtype=np.int32))
        print(f'  [{domain_name}] class {lbl} ({CLASSES[lbl]}): {len(f)} pixels')

    if not all_feats:
        return (np.zeros((0, 1), dtype=np.float32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32))

    all_feats  = np.concatenate(all_feats,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    is_source  = int(domain_name == 'source')
    all_domain = np.full(len(all_feats), is_source, dtype=np.int32)
    # domain: 1=source, 0=target
    return all_feats, all_labels, all_domain


# ===================== 绘图（需求2/3/4）=====================

def plot_tsne(tsne_xy, all_labels, all_domain, save_path):
    """
    需求2：去掉坐标轴标签、标题、图例，只保留方框和点
    需求3：源域用大三角(△)，目标域用小圆(●)，点适当放大
    需求1：颜色严格按 PALETTE
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # ---------- 先画目标域（底层），再画源域（顶层）----------
    for domain_val, marker, size, alpha, zorder in [
        (0, 'o', 40,  1.0, 2),   # 目标域：圆，偏小，半透明，底层
        (1, '^', 80,  1.0, 3),   # 源域：三角，偏大，不透明，顶层
    ]:
        mask_dom = (all_domain == domain_val)
        for lbl in range(len(CLASSES)):
            mask = mask_dom & (all_labels == lbl)
            if mask.sum() == 0:
                continue
            pts   = tsne_xy[mask]
            color = COLORS_NORM[lbl]
            ax.scatter(pts[:, 0], pts[:, 1],
                       c=[color] * len(pts),
                       marker=marker,
                       s=size,
                       alpha=alpha,
                       linewidths=0,
                       zorder=zorder)

    # ---------- 需求2：去除所有多余元素，只保留方框 ----------
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')

    # 保留四边框（spines），去掉刻度线
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')

    # 关闭坐标轴但保留 spines 的方式
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)

    plt.tight_layout(pad=0.5)
    os.makedirs(osp.dirname(osp.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'\nt-SNE figure saved → {save_path}')
    plt.close()


# ===================== 从 config 自动解析目录 =====================

def auto_find_dirs(cfg):
    def _get(ds_cfg, *keys):
        for k in keys:
            v = ds_cfg.get(k, None)
            if v is not None:
                return v
        return None

    src_img = src_ann = tgt_img = tgt_ann = None

    val_cfg = cfg.data.get('val', None)
    if val_cfg is not None:
        root  = _get(val_cfg, 'data_root', 'root')
        i_dir = _get(val_cfg, 'img_dir')
        a_dir = _get(val_cfg, 'ann_dir')
        if root and i_dir: src_img = osp.join(root, i_dir)
        if root and a_dir: src_ann = osp.join(root, a_dir)

    test_cfg = cfg.data.get('test', None)
    if test_cfg is not None:
        root  = _get(test_cfg, 'data_root', 'root')
        i_dir = _get(test_cfg, 'img_dir')
        a_dir = _get(test_cfg, 'ann_dir')
        if root and i_dir: tgt_img = osp.join(root, i_dir)
        if root and a_dir: tgt_ann = osp.join(root, a_dir)

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

    backbone = model.module.backbone_s

    # ---- 确定数据目录 ----
    src_img_dir = args.src_img_dir
    src_ann_dir = args.src_ann_dir
    tgt_img_dir = args.tgt_img_dir
    tgt_ann_dir = args.tgt_ann_dir

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
                f'请通过 --src-img-dir / --src-ann-dir / '
                f'--tgt-img-dir / --tgt-ann-dir 手动指定')

    img_size = tuple(args.img_size)

    # ---- 收集文件对 ----
    print('Collecting source file pairs...')
    src_pairs = collect_file_pairs(src_img_dir, src_ann_dir,
                                   args.img_suffix, args.ann_suffix)
    print('Collecting target file pairs...')
    tgt_pairs = collect_file_pairs(tgt_img_dir, tgt_ann_dir,
                                   args.img_suffix, args.ann_suffix)

    if not src_pairs: raise RuntimeError(f'源域无有效图像对: {src_img_dir}')
    if not tgt_pairs: raise RuntimeError(f'目标域无有效图像对: {tgt_img_dir}')

    # ---- 提取特征 ----
    print(f'\n[Source] extracting (max {args.num_images} images)...')
    src_feats, src_labels, src_domain = extract_features(
        model, backbone, src_pairs,
        num_images       = args.num_images,
        max_pixels_per_class = args.max_pixels,
        num_classes      = SOURCE_CLASS_NUM,
        img_size         = img_size,
        device           = device,
        domain_name      = 'source')

    print(f'\n[Target] extracting (max {args.num_images} images)...')
    tgt_feats, tgt_labels, tgt_domain = extract_features(
        model, backbone, tgt_pairs,
        num_images       = args.num_images,
        max_pixels_per_class = args.max_pixels,
        num_classes      = TARGET_CLASS_NUM,
        img_size         = img_size,
        device           = device,
        domain_name      = 'target')

    print(f'\nSource: {src_feats.shape}  Target: {tgt_feats.shape}')

    if src_feats.shape[0] == 0 or tgt_feats.shape[0] == 0:
        raise RuntimeError('特征提取结果为空，请检查数据目录和标签值范围。')

    all_feats  = np.concatenate([src_feats,  tgt_feats],  axis=0)
    all_labels = np.concatenate([src_labels, tgt_labels], axis=0)
    all_domain = np.concatenate([src_domain, tgt_domain], axis=0)
    print(f'Total: {len(all_feats)} samples, dim={all_feats.shape[1]}')

    # ---- 需求4：t-SNE 参数调优 ----
    # 高 perplexity + 多迭代 + early_exaggeration 放大 → 类内紧凑、类间分散
    perp = min(args.perplexity, len(all_feats) - 1)
    print(f'Running t-SNE  perplexity={perp}  max_iter={args.tsne_iter} ...')
    tsne = TSNE(
        n_components      = 2,
        perplexity        = perp,
        n_iter          = args.tsne_iter,
        early_exaggeration= 24,    # 默认12，增大→类间距离更大
        learning_rate     = 'auto',
        init              = 'pca', # PCA 初始化比随机初始化稳定
        metric='cosine',
        random_state      = 42,
        verbose           = 1,
        n_jobs            = -1,    # 多核加速
    )
    tsne_xy = tsne.fit_transform(all_feats)
    print('t-SNE done.')

    # ---- 绘图 ----
    plot_tsne(tsne_xy, all_labels, all_domain, save_path=args.save_path)


if __name__ == '__main__':
    main()