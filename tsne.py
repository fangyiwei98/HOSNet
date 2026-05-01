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
    [0,   0,   0  ],
    [0,   0,   255],
    [0,   255, 255],
    [0,   255, 0  ],
    [255, 255, 0  ],
    [255, 0,   0  ],
]
COLORS_NORM = [[r/255., g/255., b/255.] for r, g, b in PALETTE]

SOURCE_CLASS_NUM = 5   # 源域无 clutter
TARGET_CLASS_NUM = 6   # 目标域含 clutter

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
    parser.add_argument('--perplexity',  type=float, default=30.0)
    parser.add_argument('--tsne-iter',   type=int,   default=1000)
    parser.add_argument('--tsne-lr',     type=float, default=200.0)
    return parser.parse_args()


# ===================== Hook 挂载点自动探测 =====================

# 按优先级排列的候选属性名：
# 越靠前越靠近分类层（语义最强），越靠后越靠近输入
_CANDIDATE_ATTRS = [
    'sep_bottleneck',   # DepthwiseSeparableASPPHead / DecoupledOSHead
    'bottleneck',       # ASPPHead / OCRHead
    'fusion_conv',      # SegFormerHead
    'linear_fuse',      # SegFormerHead (mmseg 早期版本)
    'image_pool',       # 兜底：ASPP 全局池化层
]


def find_hook_target(head, head_name='head'):
    """
    按 _CANDIDATE_ATTRS 优先级自动找到 head 中可挂 hook 的 nn.Module。

    返回 (attr_name, module)，找不到则返回 (None, None)。
    """
    for attr in _CANDIDATE_ATTRS:
        if hasattr(head, attr):
            module = getattr(head, attr)
            if isinstance(module, torch.nn.Module):
                print(f'  [Hook] {head_name} → .{attr}  '
                      f'({type(module).__name__})')
                return attr, module

    # 找不到任何候选 → 打印 head 所有直接子 module，供用户诊断
    print(f'  [WARN] {head_name}: 未找到已知候选属性，'
          f'列出所有直接子 module：')
    for name, mod in head.named_children():
        print(f'         .{name}  ({type(mod).__name__})')
    return None, None


# ===================== Hook =====================

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
            # 取最后一个 Tensor（部分 Sequential 返回 tuple）
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


# ===================== 数据读取工具 =====================

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
    return torch.from_numpy(img).unsqueeze(0)   # [1, 3, H, W]


def load_label(ann_path, target_hw):
    lbl = np.array(Image.open(ann_path))
    if lbl.ndim == 3:
        lbl = lbl[:, :, 0]
    lbl = lbl.astype(np.int32)
    lbl_img = Image.fromarray(lbl.astype(np.uint8))
    lbl_img = lbl_img.resize((target_hw[1], target_hw[0]), Image.NEAREST)
    return np.array(lbl_img, dtype=np.int32)


# ===================== 推理触发 =====================

def run_forward(model_module, img_tensor, domain):
    """
    触发对应 head 的前向，使 hook 捕获中间特征。

    domain='source' → decode_head_s
    domain='target' → decode_head_t
    """
    F = model_module.forward_backbone(model_module.backbone_s, img_tensor)

    if domain == 'source':
        # decode_head_s 通常是 DepthwiseSeparableASPPHead
        _ = model_module.decode_head_s(F)
    else:
        # decode_head_t 通常是 DecoupledOSHead
        if hasattr(model_module.decode_head_t, 'forward_decoupled'):
            _ = model_module.decode_head_t.forward_decoupled(F)
        else:
            _ = model_module.decode_head_t(F)


# ===================== 特征提取 =====================

def extract_features(model, hook,
                     file_pairs, num_images, max_pixels_per_class,
                     num_classes, img_size, device, domain):
    """
    遍历图像，推理后从 hook 取特征，按 GT 标签采样像素。
    """
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

        feat = hook.output   # [1, C, fH, fW]
        if feat is None:
            print(f'\n  [WARN] no feature captured for {img_path}')
            continue

        # 确保是 4-D
        if feat.dim() == 3:
            feat = feat.unsqueeze(0)
        if feat.dim() != 4:
            print(f'\n  [WARN] unexpected feat shape {feat.shape}, skip')
            continue

        _, C, fH, fW = feat.shape
        feat_np = feat[0].permute(1, 2, 0).reshape(-1, C).numpy()   # [fH*fW, C]

        # 将 GT 缩放到特征图分辨率
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


# ===================== 绘图 =====================

def plot_tsne(tsne_xy, all_labels, all_domain, save_path):
    """
    源域：大三角(▲)，目标域：小圆(●)
    颜色按 PALETTE，无坐标/标题/图例，保留方框
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 目标域底层，源域顶层
    for domain_val, marker, size, alpha, zorder in [
        (0, 'o', 35,  0.55, 2),   # target
        (1, '^', 70,  0.90, 3),   # source
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

    ax.set_xticks([]);  ax.set_yticks([])
    ax.set_xlabel('');  ax.set_ylabel('');  ax.set_title('')
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')

    plt.tight_layout(pad=0.5)
    os.makedirs(osp.dirname(osp.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'\nt-SNE figure saved → {save_path}')
    plt.close()


# ===================== main =====================

def main():
    args = parse_args()
    cfg  = mmcv.Config.fromfile(args.config)
    setup_multi_processes(cfg)

    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    # ── 构建 & 加载模型 ───────────────────────────────────────────
    cfg.model.pretrained = None
    cfg.model.train_cfg  = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = revert_sync_batchnorm(model)
    model = MMDataParallel(model, device_ids=[args.gpu_id])
    model.to(device)
    model.eval()

    m = model.module   # 真实 OSNet 实例

    # ── 打印 head 类型（帮助确认 config 是否正确加载）────────────
    print(f'\ndecode_head_s type: {type(m.decode_head_s).__name__}')
    print(f'decode_head_t type: {type(m.decode_head_t).__name__}')

    # ── 自动探测 hook 挂载点 ─────────────────────────────────────
    print('\n[Hook detection]')
    src_attr, src_module = find_hook_target(m.decode_head_s, 'decode_head_s')
    tgt_attr, tgt_module = find_hook_target(m.decode_head_t, 'decode_head_t')

    if src_module is None:
        raise RuntimeError(
            'decode_head_s 中找不到合适的 hook 挂载点，'
            '请在 _CANDIDATE_ATTRS 中添加对应属性名。')
    if tgt_module is None:
        raise RuntimeError(
            'decode_head_t 中找不到合适的 hook 挂载点，'
            '请在 _CANDIDATE_ATTRS 中添加对应属性名。')

    # ── 数据目录验证 ──────────────────────────────────────────────
    for d, name in [(args.src_img_dir, 'src_img'),
                    (args.src_ann_dir, 'src_ann'),
                    (args.tgt_img_dir, 'tgt_img'),
                    (args.tgt_ann_dir, 'tgt_ann')]:
        if d is None or not osp.isdir(d):
            raise FileNotFoundError(f'{name} 目录不存在: {d}')

    img_size = tuple(args.img_size)

    # ── 收集文件对 ────────────────────────────────────────────────
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

    # ── 提取源域特征 ──────────────────────────────────────────────
    print(f'\n[Source] using decode_head_s.{src_attr} as feature layer '
          f'(max {args.num_images} images)...')
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

    # ── 提取目标域特征 ────────────────────────────────────────────
    print(f'\n[Target] using decode_head_t.{tgt_attr} as feature layer '
          f'(max {args.num_images} images)...')
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

    # ── 特征维度对齐检查 ─────────────────────────────────────────
    # 两个 head 的特征维度可能不同（如 DeepLab=256，SegFormer=768）
    # t-SNE 可以直接处理不同维度拼接，但语义上不可比
    # 推荐做法：分别降维再合并，或只比较同维度特征
    if src_feats.shape[1] != tgt_feats.shape[1]:
        print(f'\n[WARN] 源域特征维度 {src_feats.shape[1]} ≠ '
              f'目标域特征维度 {tgt_feats.shape[1]}')
        print('       将分别做 PCA 降至 128 维后再合并进行 t-SNE ...')
        from sklearn.decomposition import PCA
        pca_dim = 128
        pca_src = PCA(n_components=pca_dim, random_state=42)
        pca_tgt = PCA(n_components=pca_dim, random_state=42)
        src_feats = pca_src.fit_transform(src_feats)
        tgt_feats = pca_tgt.fit_transform(tgt_feats)
        print(f'       PCA 后: src {src_feats.shape}, tgt {tgt_feats.shape}')

    # ── 拼接 ─────────────────────────────────────────────────────
    all_feats  = np.concatenate([src_feats,  tgt_feats],  axis=0)
    all_labels = np.concatenate([src_labels, tgt_labels], axis=0)
    all_domain = np.concatenate([src_domain, tgt_domain], axis=0)
    print(f'Total: {len(all_feats)} samples, feat_dim={all_feats.shape[1]}')

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

    # ── 绘图 ─────────────────────────────────────────────────────
    plot_tsne(tsne_xy, all_labels, all_domain, save_path=args.save_path)


if __name__ == '__main__':
    main()