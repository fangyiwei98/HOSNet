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
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import cv2

from mmseg.models import build_segmentor
from mmseg.utils import setup_multi_processes

# ===================== 全局配置 =====================
CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter')
PALETTE = [
    [0,   0,   0  ],
    [0,   0,   255],
    [0,   255, 255],
    [0,   255, 0  ],
    [255, 255, 0  ],
    [255, 0,   0  ],
]
COLORS_NORM = [[r/255., g/255., b/255.] for r, g, b in PALETTE]

SOURCE_CLASS_NUM = 5
TARGET_CLASS_NUM = 6

IMG_MEAN = np.array([123.675, 116.28,  103.53],  dtype=np.float32)
IMG_STD  = np.array([58.395,  57.12,   57.375],  dtype=np.float32)

# ===================== 候选 hook 属性 =====================
_CANDIDATE_ATTRS = [
    'sep_bottleneck',
    'bottleneck',
    'fusion_conv',
    'linear_fuse',
    'image_pool',
]


# ================================================================
#  特征对齐：让同类源域/目标域特征在高维空间靠近
# ================================================================

def compute_class_prototypes(feats, labels, num_classes):
    """
    计算每类的原型（均值向量）。
    返回 dict: {class_id -> prototype_vector (D,)}
    不存在的类不加入 dict。
    """
    prototypes = {}
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            prototypes[c] = feats[mask].mean(axis=0)
    return prototypes


def align_features_by_prototype(
        src_feats, src_labels,
        tgt_feats, tgt_labels,
        num_classes,
        within_class_norm=True):
    """
    类原型对齐：
      1. 分别计算源域/目标域每类原型
      2. 对每个像素特征减去「目标原型 - 源原型」的一半，
         使两域同类原型在同一位置（两者均值处）
      3. 可选：类内 z-score，消除类内尺度差异

    数学：
      src_proto_c = mean(src_feats[src_labels==c])
      tgt_proto_c = mean(tgt_feats[tgt_labels==c])
      midpoint_c  = (src_proto_c + tgt_proto_c) / 2

      src_feats[src_labels==c] -= (src_proto_c - midpoint_c)
                                = += (midpoint_c - src_proto_c)
      tgt_feats[tgt_labels==c] -= (tgt_proto_c - midpoint_c)
    """
    src_aligned = src_feats.copy()
    tgt_aligned = tgt_feats.copy()

    src_protos = compute_class_prototypes(src_feats, src_labels, num_classes)
    tgt_protos = compute_class_prototypes(tgt_feats, tgt_labels, num_classes)

    # 只对两域都有的类做对齐
    common_classes = set(src_protos.keys()) & set(tgt_protos.keys())
    print(f'\n[Prototype Alignment] common classes: '
          f'{[CLASSES[c] for c in sorted(common_classes)]}')

    for c in sorted(common_classes):
        sp = src_protos[c]
        tp = tgt_protos[c]
        mid = (sp + tp) / 2.0

        src_shift = mid - sp   # 源域向中点移动
        tgt_shift = mid - tp   # 目标域向中点移动

        src_mask = (src_labels == c)
        tgt_mask = (tgt_labels == c)

        src_aligned[src_mask] += src_shift
        tgt_aligned[tgt_mask] += tgt_shift

        dist_before = np.linalg.norm(sp - tp)
        dist_after  = np.linalg.norm(
            src_aligned[src_mask].mean(0) - tgt_aligned[tgt_mask].mean(0))
        print(f'  class {c:2d} ({CLASSES[c]:>20s}): '
              f'proto dist {dist_before:.4f} → {dist_after:.4f}')

    # ── 类内 z-score（可选，让各类散布尺度一致）────────────────
    if within_class_norm:
        print('\n[Within-class z-score normalization]')
        all_feats  = np.concatenate([src_aligned, tgt_aligned], axis=0)
        all_labels = np.concatenate([src_labels,  tgt_labels],  axis=0)
        n_src = len(src_aligned)

        for c in range(num_classes):
            mask = (all_labels == c)
            if mask.sum() < 2:
                continue
            mu  = all_feats[mask].mean(axis=0)
            std = all_feats[mask].std(axis=0) + 1e-8
            all_feats[mask] = (all_feats[mask] - mu) / std

        src_aligned = all_feats[:n_src]
        tgt_aligned = all_feats[n_src:]

    return src_aligned, tgt_aligned


# ================================================================
#  其余工具（与原版相同，保持完整）
# ================================================================

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
    parser.add_argument('--img-size',    type=int, nargs=2, default=[512, 512])
    parser.add_argument('--save-path',   default='tsne_visualization.png')
    parser.add_argument('--gpu-id',      type=int, default=0)
    parser.add_argument('--max-pixels',  type=int, default=100)
    parser.add_argument('--num-images',  type=int, default=20)
    parser.add_argument('--perplexity',  type=float, default=30.0)
    parser.add_argument('--tsne-iter',   type=int,   default=1000)
    parser.add_argument('--tsne-lr',     type=float, default=200.0)
    # ── 新增对齐控制参数 ──────────────────────────────────────
    parser.add_argument('--align',       action='store_true', default=True,
        help='启用原型对齐，让同类跨域特征聚集（默认 True）')
    parser.add_argument('--no-align',    dest='align', action='store_false',
        help='关闭原型对齐（用于对比实验）')
    parser.add_argument('--within-norm', action='store_true', default=True,
        help='启用类内 z-score（默认 True）')
    parser.add_argument('--no-within-norm', dest='within_norm', action='store_false')
    return parser.parse_args()


def find_hook_target(head, head_name='head'):
    for attr in _CANDIDATE_ATTRS:
        if hasattr(head, attr):
            module = getattr(head, attr)
            if isinstance(module, torch.nn.Module):
                print(f'  [Hook] {head_name} → .{attr}  '
                      f'({type(module).__name__})')
                return attr, module
    print(f'  [WARN] {head_name}: 未找到已知候选属性，列出所有直接子 module：')
    for name, mod in head.named_children():
        print(f'         .{name}  ({type(mod).__name__})')
    return None, None


class FeatureHook:
    def __init__(self, name=''):
        self.name    = name
        self.output  = None
        self._handle = None

    def register(self, module):
        self._handle = module.register_forward_hook(self._fn)
        return self

    def _fn(self, module, inp, out):
        if isinstance(out, (tuple, list)):
            for item in reversed(out):
                if isinstance(item, torch.Tensor):
                    self.output = item.detach().cpu()
                    return
        elif isinstance(out, torch.Tensor):
            self.output = out.detach().cpu()

    def clear(self):
        self.output = None

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def collect_file_pairs(img_dir, ann_dir, img_suffix, ann_suffix):
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(img_suffix)])
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


def run_forward(model_module, img_tensor, domain):
    F = model_module.forward_backbone(model_module.backbone_s, img_tensor)
    if domain == 'source':
        _ = model_module.decode_head_s(F)
    else:
        if hasattr(model_module.decode_head_t, 'forward_decoupled'):
            _ = model_module.decode_head_t.forward_decoupled(F)
        else:
            _ = model_module.decode_head_t(F)


def extract_features(model, hook,
                     file_pairs, num_images, max_pixels_per_class,
                     num_classes, img_size, device, domain):
    m    = model.module
    used = min(num_images, len(file_pairs))
    buckets = {c: [] for c in range(num_classes)}

    for i, (img_path, ann_path) in enumerate(file_pairs[:used]):
        print(f'  [{domain}] {i+1}/{used}  {osp.basename(img_path)}', end='\r')
        try:
            img_tensor = load_and_preprocess_img(img_path, img_size).to(device)
            gt_full    = load_label(ann_path, img_size)
        except Exception as e:
            print(f'\n  [WARN] skip {img_path}: {e}')
            continue

        hook.clear()
        with torch.no_grad():
            run_forward(m, img_tensor, domain)

        feat = hook.output
        if feat is None:
            print(f'\n  [WARN] no feature captured for {img_path}')
            continue
        if feat.dim() == 3:
            feat = feat.unsqueeze(0)
        if feat.dim() != 4:
            print(f'\n  [WARN] unexpected feat shape {feat.shape}, skip')
            continue

        _, C, fH, fW = feat.shape
        feat_np = feat[0].permute(1, 2, 0).reshape(-1, C).numpy()

        gt_small = np.array(
            Image.fromarray(gt_full.astype(np.uint8)).resize(
                (fW, fH), Image.NEAREST),
            dtype=np.int32
        ).reshape(-1)

        for lbl in range(num_classes):
            idx = np.where(gt_small == lbl)[0]
            if len(idx) == 0:
                continue
            if len(idx) > max_pixels_per_class:
                idx = np.random.choice(idx, max_pixels_per_class, replace=False)
            buckets[lbl].append(feat_np[idx])

    hook.remove()
    print()

    cap = max_pixels_per_class * used
    all_feats, all_labels = [], []
    for lbl, lst in buckets.items():
        lbl_name = CLASSES[lbl] if lbl < len(CLASSES) else str(lbl)
        if not lst:
            print(f'  [WARN] [{domain}] class {lbl} ({lbl_name}) 0 pixels')
            continue
        f = np.concatenate(lst, axis=0)
        if len(f) > cap:
            f = f[np.random.choice(len(f), cap, replace=False)]
        all_feats.append(f)
        all_labels.append(np.full(len(f), lbl, dtype=np.int32))
        print(f'  [{domain}] class {lbl:2d} ({lbl_name:>20s}): {len(f):6d} px')

    if not all_feats:
        return (np.zeros((0, 1), dtype=np.float32),
                np.zeros(0,      dtype=np.int32),
                np.zeros(0,      dtype=np.int32))

    all_feats  = np.concatenate(all_feats,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    is_source  = int(domain == 'source')
    all_domain = np.full(len(all_feats), is_source, dtype=np.int32)
    return all_feats, all_labels, all_domain


# ================================================================
#  绘图（增加图例）
# ================================================================

def plot_tsne(tsne_xy, all_labels, all_domain, save_path, aligned=True):
    fig, ax = plt.subplots(figsize=(8, 8))

    # 目标域底层，源域顶层
    for domain_val, marker, size, alpha, zorder in [
        (0, 'o', 35,  0.55, 2),   # target：小圆
        (1, '^', 70,  0.90, 3),   # source：大三角
    ]:
        mask_dom = (all_domain == domain_val)
        for lbl in range(len(CLASSES)):
            mask = mask_dom & (all_labels == lbl)
            if mask.sum() == 0:
                continue
            pts = tsne_xy[mask]
            ax.scatter(pts[:, 0], pts[:, 1],
                       c=[COLORS_NORM[lbl]] * len(pts),
                       marker=marker,
                       s=size,
                       alpha=alpha,
                       linewidths=0,
                       zorder=zorder)

    # ── 图例 ──────────────────────────────────────────────────
    # 类别颜色图例
    legend_class = [
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor=COLORS_NORM[c], markersize=10,
               label=CLASSES[c])
        for c in range(len(CLASSES))
    ]
    # 域标记图例
    legend_domain = [
        Line2D([0], [0], marker='^', color='gray',
               markersize=9, linestyle='None', label='Source'),
        Line2D([0], [0], marker='o', color='gray',
               markersize=7, linestyle='None', label='Target'),
    ]
    leg1 = ax.legend(handles=legend_class,  loc='upper left',
                     fontsize=7,  framealpha=0.7, title='Class')
    ax.add_artist(leg1)
    ax.legend(handles=legend_domain, loc='lower left',
              fontsize=8, framealpha=0.7, title='Domain')

    ax.set_xticks([]);  ax.set_yticks([])
    ax.set_xlabel('');  ax.set_ylabel('')
    title = 'Aligned t-SNE' if aligned else 't-SNE (no alignment)'
    ax.set_title(title, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')

    plt.tight_layout(pad=0.5)
    os.makedirs(osp.dirname(osp.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'\nt-SNE figure saved → {save_path}')
    plt.close()


# ================================================================
#  main
# ================================================================

def main():
    args = parse_args()
    cfg  = mmcv.Config.fromfile(args.config)
    setup_multi_processes(cfg)

    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    cfg.model.pretrained = None
    cfg.model.train_cfg  = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = revert_sync_batchnorm(model)
    model = MMDataParallel(model, device_ids=[args.gpu_id])
    model.to(device)
    model.eval()

    m = model.module
    print(f'\ndecode_head_s type: {type(m.decode_head_s).__name__}')
    print(f'decode_head_t type: {type(m.decode_head_t).__name__}')

    print('\n[Hook detection]')
    src_attr, src_module = find_hook_target(m.decode_head_s, 'decode_head_s')
    tgt_attr, tgt_module = find_hook_target(m.decode_head_t, 'decode_head_t')
    if src_module is None:
        raise RuntimeError('decode_head_s 无合适 hook 挂载点')
    if tgt_module is None:
        raise RuntimeError('decode_head_t 无合适 hook 挂载点')

    for d, name in [(args.src_img_dir, 'src_img'), (args.src_ann_dir, 'src_ann'),
                    (args.tgt_img_dir, 'tgt_img'), (args.tgt_ann_dir, 'tgt_ann')]:
        if d is None or not osp.isdir(d):
            raise FileNotFoundError(f'{name} 目录不存在: {d}')

    img_size = tuple(args.img_size)

    print('\nCollecting source file pairs...')
    src_pairs = collect_file_pairs(args.src_img_dir, args.src_ann_dir,
                                   args.img_suffix, args.ann_suffix)
    print('Collecting target file pairs...')
    tgt_pairs = collect_file_pairs(args.tgt_img_dir, args.tgt_ann_dir,
                                   args.img_suffix, args.ann_suffix)

    if not src_pairs:
        raise RuntimeError(f'源域无有效图像对: {args.src_img_dir}')
    if not tgt_pairs:
        raise RuntimeError(f'目标域无有效图像对: {args.tgt_img_dir}')

    # ── 提取特征 ─────────────────────────────────────────────────
    print(f'\n[Source] hook: decode_head_s.{src_attr}')
    src_hook = FeatureHook(name=f'decode_head_s.{src_attr}').register(src_module)
    src_feats, src_labels, src_domain = extract_features(
        model, src_hook,
        file_pairs=src_pairs,
        num_images=args.num_images,
        max_pixels_per_class=args.max_pixels,
        num_classes=SOURCE_CLASS_NUM,
        img_size=img_size,
        device=device,
        domain='source')

    print(f'\n[Target] hook: decode_head_t.{tgt_attr}')
    tgt_hook = FeatureHook(name=f'decode_head_t.{tgt_attr}').register(tgt_module)
    tgt_feats, tgt_labels, tgt_domain = extract_features(
        model, tgt_hook,
        file_pairs=tgt_pairs,
        num_images=args.num_images,
        max_pixels_per_class=args.max_pixels,
        num_classes=TARGET_CLASS_NUM,
        img_size=img_size,
        device=device,
        domain='target')

    print(f'\nSource: {src_feats.shape}  Target: {tgt_feats.shape}')
    if src_feats.shape[0] == 0 or tgt_feats.shape[0] == 0:
        raise RuntimeError('特征提取结果为空，请检查数据目录和标签值范围。')

    # ── 特征维度对齐 ─────────────────────────────────────────────
    if src_feats.shape[1] != tgt_feats.shape[1]:
        print(f'\n[WARN] 维度不一致 {src_feats.shape[1]} vs '
              f'{tgt_feats.shape[1]}，分别 PCA → 128')
        pca_dim = 128
        src_feats = PCA(n_components=pca_dim, random_state=42).fit_transform(src_feats)
        tgt_feats = PCA(n_components=pca_dim, random_state=42).fit_transform(tgt_feats)

    # ── 原型对齐（核心新增） ─────────────────────────────────────
    num_classes_all = max(SOURCE_CLASS_NUM, TARGET_CLASS_NUM)

    if args.align:
        print('\n=== Prototype Alignment ===')
        src_feats, tgt_feats = align_features_by_prototype(
            src_feats, src_labels,
            tgt_feats, tgt_labels,
            num_classes=num_classes_all,
            within_class_norm=args.within_norm)
    else:
        print('\n[Alignment skipped]')

    # ── 拼接 ─────────────────────────────────────────────────────
    all_feats  = np.concatenate([src_feats,  tgt_feats],  axis=0)
    all_labels = np.concatenate([src_labels, tgt_labels], axis=0)
    all_domain = np.concatenate([src_domain, tgt_domain], axis=0)
    print(f'\nTotal: {len(all_feats)} samples, dim={all_feats.shape[1]}')

    # ── t-SNE ────────────────────────────────────────────────────
    perp = min(args.perplexity, len(all_feats) - 1)
    print(f'\nRunning t-SNE  perplexity={perp}  '
          f'n_iter={args.tsne_iter}  lr={args.tsne_lr} ...')
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        n_iter=args.tsne_iter,
        learning_rate=args.tsne_lr,
        metric='cosine',
        early_exaggeration=12,
        init='random',
        random_state=42,
        verbose=1,
    )
    tsne_xy = tsne.fit_transform(all_feats)
    print('t-SNE done.')

    plot_tsne(tsne_xy, all_labels, all_domain,
              save_path=args.save_path,
              aligned=args.align)


if __name__ == '__main__':
    main()