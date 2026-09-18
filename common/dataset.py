"""PyTorch Dataset：图片 + RSML（茎/叶）+ labelme json（地上部/种茎）三通道真值。

数据布局**两种都认**（先扁平，再嵌套）：

    扁平（当前数据集就是这样，图片/标注同层）：
        <data_dir>/<名>.jpg|png      原图
        <data_dir>/<名>.rsml         茎与叶的折线（必须有，配对依据）
        <data_dir>/<名>.json         labelme 标注（可缺）

    嵌套（root_model 那种布局，兼容用）：
        <data_dir>/images/<名>.<ext>
        <data_dir>/labels/roots/<名>.rsml
        <data_dir>/labels/other/<名>.json

目标张量 (3, H, W) 的通道顺序见 config.CLASS_NAMES：
    0 = shoot   幼苗的茎：rsml 的 1.1（基部 → 叶鞘开口顶端），折线画线
    1 = above   地上部：json 的 above_ground 矩形，填充
    2 = sett    种茎横截面：json 的 stem polygon，填充

三张掩码都在**模型输入分辨率**上直接绘制（坐标先按比例换算），比「原图分辨率画好再
缩放」快约 200 倍，且细线不会被最近邻降采样漏掉（见 common/gt_mask.py 说明）。

与 root_model/common/dataset.py 的差别：那边是 root/stem/check 三通道、只认嵌套布局、
根空壳有专门的告警；这边是 shoot/above/sett、扁平优先、茎与叶要分开挑。
增强函数（_affine_* / _jitter / _corner_fill）是逐字照抄过来的。
"""
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import config
from common import gt_mask, image_io
from common.labelme import parse_labels
from common.rsml_parse import parse_rsml

# 通道序号（与 config.CLASS_NAMES 一致）
CH_SHOOT, CH_ABOVE, CH_SETT = 0, 1, 2

# rsml 里哪条折线算「茎」：ID 形如 "1.1" / "2.1"（深度 1 的主根），叶子是 "1.1.1"、"1.1.2"…
_PRIMARY_ID = re.compile(r"^\d+\.1$")


def _label_dirs(data_dir: Path):
    """返回 (图片目录, 标注目录) —— 扁平优先，找不到图片再回退嵌套布局。

    扁平布局下两者都是 data_dir 本身。
    """
    sub = data_dir / config.IMAGES_SUBDIR
    if sub.is_dir():                      # 嵌套布局
        return sub, data_dir / "labels"
    return data_dir, data_dir             # 扁平布局（当前数据集）


def discover_pairs(data_dir, image_exts=None) -> list:
    """返回 [(stem, 图片路径, rsml路径, json路径或None), ...]：图片必须有同名 .rsml 配对。

    **找不到 rsml 的图片会被跳过**（不是报错，便于混放）。
    注意 root_model 的版本只认嵌套布局，喂扁平目录会**静默返回空列表** —— 这里修掉了。
    """
    if image_exts is None:
        image_exts = config.IMAGE_EXTS
    data_dir = Path(data_dir)
    img_dir, lab_dir = _label_dirs(data_dir)
    if not img_dir.is_dir():
        return []
    # 嵌套布局里 rsml 与 json 分两个子目录；扁平布局里都在 lab_dir 本身
    roots_dir = (lab_dir / "roots") if (lab_dir / "roots").is_dir() else lab_dir
    other_dir = (lab_dir / "other") if (lab_dir / "other").is_dir() else lab_dir

    pairs = []
    for img_path in sorted(p for p in img_dir.iterdir()
                           if p.suffix.lower() in image_exts):
        rsml_path = roots_dir / f"{img_path.stem}.rsml"
        if not rsml_path.exists():
            continue
        json_path = other_dir / f"{img_path.stem}.json"
        pairs.append((img_path.stem, img_path, rsml_path,
                      json_path if json_path.exists() else None))
    return pairs


def plant_key(name: str) -> str:
    """图片名 -> 植株标识，用于「整株进出」的数据划分。

    'plant_C001-1_20241229CK' -> 'C001-1'
    'root_S062-1_20251116ST'  -> 'S062-1'

    与 root_model 的同名函数语义一致：先剥前缀，再去掉末尾的「日期+批次」记号。
    """
    s = str(name).replace(" ", "")
    for pre in ("plant_", "root_"):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    head, sep, tail = s.rpartition("_")
    if sep and len(tail) >= 6 and any(ch.isdigit() for ch in tail):
        s = head
    return s


# ------------------------- 数据增强（逐字照抄自 root_model/common/dataset.py） -------------------------

def _corner_fill(img: np.ndarray, patch: int = 32) -> tuple:
    """取图像四角小块的均值颜色，作为旋转增强的填充色（避免黑边假象）。"""
    h, w = img.shape[:2]
    corners = [img[0:patch, 0:patch], img[0:patch, w - patch:w],
               img[h - patch:h, 0:patch], img[h - patch:h, w - patch:w]]
    mean = np.mean(np.concatenate(corners).reshape(-1, 3), axis=0)
    return (int(mean[0]), int(mean[1]), int(mean[2]))


def _affine_matrix(angle: float, scale: float, w: int, h: int) -> tuple:
    """绕画布中心「先缩放、再旋转」的仿射矩阵（PIL AFFINE 的输出->输入方向）。"""
    rad = math.radians(angle)
    cos, sin = math.cos(rad), math.sin(rad)
    cx, cy = w / 2.0, h / 2.0
    a, b = cos / scale, sin / scale
    d, e = -sin / scale, cos / scale
    return (a, b, cx - a * cx - b * cy, d, e, cy - d * cx - e * cy)


def _affine_rgb(arr: np.ndarray, angle: float, scale: float, fill: tuple) -> np.ndarray:
    h, w = arr.shape[:2]
    im = Image.fromarray(arr).transform(
        (w, h), Image.AFFINE, _affine_matrix(angle, scale, w, h),
        resample=Image.Resampling.BILINEAR, fillcolor=fill)
    return np.asarray(im, dtype=np.uint8)


def _affine_mask(mask: np.ndarray, angle: float, scale: float) -> np.ndarray:
    h, w = mask.shape[:2]
    im = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).transform(
        (w, h), Image.AFFINE, _affine_matrix(angle, scale, w, h),
        resample=Image.Resampling.NEAREST, fillcolor=0)
    return np.asarray(im, dtype=np.uint8) > 127


def _jitter(img: np.ndarray) -> np.ndarray:
    """亮度/对比度/饱和度抖动（模拟不同批次的光照与白平衡差异）。"""
    out = img.astype(np.float32)
    b = random.uniform(*config.AUG_BRIGHTNESS)
    c = random.uniform(*config.AUG_CONTRAST)
    out = (out * b - 127.5) * c + 127.5
    out = np.clip(out, 0, 255).astype(np.uint8)
    s = random.uniform(*config.AUG_SATURATION)
    if abs(s - 1.0) > 0.01:
        from PIL import ImageEnhance
        out = np.asarray(ImageEnhance.Color(Image.fromarray(out)).enhance(s),
                         dtype=np.uint8)
    return out


# ------------------------- 真值掩码 -------------------------

_warned_no_shoot = set()


def _warn_no_shoot(rsml_path):
    """同一文件只吵一次。rsml 里没有 1.1 就画不出茎 → 茎轴无从谈起，株高也就不用算了。"""
    key = str(rsml_path)
    if key in _warned_no_shoot:
        return
    _warned_no_shoot.add(key)
    print(f"[提示] {Path(rsml_path).name} 里没有 ID 形如 1.1 的折线（茎），"
          f"该图的 shoot 通道将被屏蔽。")


def build_target_masks(rsml_path, json_path, orig_size, target_size, mask_width):
    """画三通道真值掩码，返回 (masks[h,w,3] bool, chan_valid[3] float)。

    test.py / inference.py 复用同一份实现，保证「训练真值」与「评测真值」口径一致。

    缺标注时：对应通道置空并标为「无效」——训练时该通道的损失会被屏蔽，
    避免模型学成「这里没有东西」。
    """
    w1, h1 = target_size
    masks = np.zeros((h1, w1, 3), dtype=bool)
    valid = np.zeros(3, dtype=np.float32)

    # ---- shoot：rsml 里 ID 形如 1.1 的那条折线 ----
    roots = parse_rsml(rsml_path)
    stems = [r for r in roots if _PRIMARY_ID.match(str(r.root_id))]
    if stems:
        line_w = gt_mask.target_line_width(mask_width, orig_size, target_size)
        polys = [gt_mask.scale_points(r.points, orig_size, target_size)
                 for r in stems if len(r.points) >= 2]
        masks[:, :, CH_SHOOT] = gt_mask.draw_polylines_at(polys, target_size, line_w)
        valid[CH_SHOOT] = 1.0
    else:
        _warn_no_shoot(rsml_path)

    # ---- above + sett：labelme json ----
    if json_path is not None:
        lab = parse_labels(json_path, image_size=orig_size)
        # **用多边形画，不要用外接框**：above_ground 要标成沿茎轴的旋转矩形，
        # 压成外接框会把旋转信息丢掉、株高就重新带上倾角偏差（实测最多 +19%）。
        # 见 common/labelme.py 模块 docstring。
        if lab.above_polys:
            masks[:, :, CH_ABOVE] = gt_mask.draw_polygons_at(
                [gt_mask.scale_points(p, orig_size, target_size) for p in lab.above_polys],
                target_size)
            valid[CH_ABOVE] = 1.0
        if lab.setts:
            masks[:, :, CH_SETT] = gt_mask.draw_polygons_at(
                [gt_mask.scale_points(p, orig_size, target_size) for p in lab.setts],
                target_size)
            valid[CH_SETT] = 1.0
    return masks, valid


class PlantDataset(Dataset):
    """逐项返回 (img[3,H,W] float32 0~1, gt[3,H,W] float32 0/1, name, chan_valid[3])。

    构造时完成解码/画掩码/缩放（较慢），训练时仅做轻量增强。
    chan_valid 标记该图哪些通道有真值（缺 json / 缺 1.1 时为 0），训练侧据此屏蔽损失。
    """

    def __init__(self, data_dir, names=None, max_side=1024, stride=16,
                 mask_width=None, augment=False, seed=0):
        if mask_width is None:
            mask_width = config.MASK_LINE_WIDTH   # 显式取 config，别用隐式默认值
        self.augment = augment
        self.data_dir = Path(data_dir)
        pairs = discover_pairs(self.data_dir)
        if not pairs:
            raise RuntimeError(
                f"{self.data_dir} 里没有找到任何「图片 + 同名 .rsml」配对。"
                f"检查：图片扩展名是否在 {sorted(config.IMAGE_EXTS)} 里、"
                f".rsml 是否与图片同名同目录（或放在 labels/roots/）。")
        if names is not None:
            wanted = set(names)
            pairs = [p for p in pairs if p[0] in wanted]
            missing = sorted(wanted - {p[0] for p in pairs})
            if missing:
                print(f"[警告] 有 {len(missing)} 个名字在数据集中找不到配对: {missing[:5]}")
        self.names = [p[0] for p in pairs]

        self.items = []
        n_no_json = 0
        for name, img_path, rsml_path, json_path in pairs:
            img = image_io.load_rgb(img_path)
            h0, w0 = img.shape[:2]
            w1, h1 = image_io.target_size(w0, h0, max_side, stride)
            masks, valid = build_target_masks(rsml_path, json_path,
                                              (w0, h0), (w1, h1), mask_width)
            if valid[CH_ABOVE] == 0 or valid[CH_SETT] == 0:
                n_no_json += 1
            self.items.append({
                "name": name,
                "img": image_io.resize_rgb(img, w1, h1),   # (h1,w1,3) uint8
                "masks": masks,                            # (h1,w1,3) bool
                "valid": valid,                            # (3,) float32
                "fill": _corner_fill(img),
            })
        if n_no_json:
            print(f"[警告] {n_no_json} 张图缺 above_ground / stem 标注（json 缺或不全），"
                  f"训练时这两个通道的损失会被屏蔽。")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = it["img"]
        m = it["masks"]
        if self.augment:
            if random.random() < 0.5:
                img = np.flip(img, axis=1)
                m = np.flip(m, axis=1)
            # 旋转 + 尺度抖动一起做（同一个仿射矩阵作用于图像与三张掩码，保证对齐）
            angle = random.uniform(-config.AUG_ROTATE_DEG, config.AUG_ROTATE_DEG)
            scale = random.uniform(*config.AUG_SCALE)
            if abs(angle) > 0.3 or abs(scale - 1.0) > 0.01:
                img = _affine_rgb(img, angle, scale, it["fill"])
                m = np.stack([_affine_mask(m[:, :, c], angle, scale)
                              for c in range(m.shape[2])], axis=2)
            img = _jitter(img)
        img = np.ascontiguousarray(img)
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        y = torch.from_numpy(np.ascontiguousarray(
            m.transpose(2, 0, 1), dtype=np.float32))          # (3,H,W)
        return x, y, it["name"], torch.from_numpy(it["valid"].copy())
