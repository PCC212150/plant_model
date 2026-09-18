"""labelme 标注（与图片同名的 .json）解析：种茎横截面 + 地上部范围。

json 结构（labelme 6.x）：
    {"version","flags","shapes":[{label,shape_type,points,...}],
     "imagePath","imageData","imageHeight","imageWidth"}

本项目只用两类 label：
    stem           **种茎的横截面**（polygon）。注意：不是新长出的茎，
                   是播下去那一截甘蔗种子的切面，用来定位基部在哪一端。
    above_ground   地上部范围（rectangle）。**株高就是它沿茎轴方向的伸出量**
                   （见 common/measure.py）。

**`above_ground` 要画成「沿茎轴方向的旋转矩形」，不要轴对齐**（2026-09-18 与用户确认）：
    轴对齐矩形的"沿茎轴最远点"落在**角**上，而角上没有植株，于是株高系统性偏大，
    偏多少取决于茎的倾角 —— 实测 8 张：茎水平时 +0.6~1.7%，茎翘 31.4° 时 **+19.1%**。
    把矩形转成与茎平行后，最远点落在**边**上，而矩形是紧贴画的最远边必然碰到植株
    （数学上：对齐后矩形沿轴的长度 ≡ 植株沿轴的长度），偏差消失。
    标注成本几乎不变：还是画 4 个点（labelme 的 4 点 rectangle），只是摆正方向。

    两种都认：**4 点**（旋转矩形，按原样保留）、**2 点**（轴对齐，展开成 4 个角）。
    **4 点绝不能被压成外接框** —— 那等于把旋转信息丢掉、偏差又回来了。

解析结果统一为**原图坐标**下的矢量（点列 / 外接矩形），画掩码时按目标尺寸换算
（见 common/gt_mask.py），这样同一份标注可以按任意输入分辨率绘制。

注意：labelme 默认会把整张图 base64 塞进 imageData，解析后必须立刻丢弃，
绝不能把整个 dict 缓存下来。

**与 root_model/common/labelme.py 的差别**：那边的第二类 label 是 `check_background`
（检查范围），这边是 `above_ground`（地上部）。两边的 `stem` 都指种茎/蔗茎的横截面，
但本项目的"茎"（幼苗的茎）来自 RSML 的 1.1，叫 `shoot` —— 见 config.CLASS_NAMES。
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

SETT_LABEL = "stem"             # 种茎横截面（polygon）
ABOVE_LABEL = "above_ground"    # 地上部外接框（rectangle）


@dataclass
class PlantLabels:
    """一张图的 labelme 标注（原图坐标）。"""

    path: Path = None
    setts: list = field(default_factory=list)       # [[(x, y), ...], ...] 种茎横截面多边形
    above_polys: list = field(default_factory=list)  # [[(x, y), ...], ...] 地上部四边形
                                                    # （4 点旋转矩形原样保留；2 点轴对齐展开成 4 角）
    info: dict = field(default_factory=dict)        # 自检信息（shape 数 / 面积占比 / 告警）

    @property
    def above_rect(self) -> tuple:
        """地上部的**外接框** (x0, y0, x1, y1)。只用于报告/告警，**画掩码不要用它** ——
        它会把旋转矩形压扁，见模块 docstring。"""
        if not self.above_polys:
            return None
        pts = [p for poly in self.above_polys for p in poly]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def ok(self) -> bool:
        """两类标注都解析到了才为 True（缺任一类的图，训练时会屏蔽对应通道的损失）。"""
        return bool(self.setts) and bool(self.above_polys)


def _finite_points(points) -> list:
    """过滤非有限坐标（NaN/inf），返回 [(x, y), ...]。"""
    out = []
    for p in points or []:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            out.append((x, y))
    return out


def _polygon_area(points) -> float:
    """鞋带公式算多边形面积（用于面积占比告警，不做精确统计）。"""
    if len(points) < 3:
        return 0.0
    s = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def parse_labels(json_path, image_size=None, verbose: bool = True) -> PlantLabels:
    """解析 labelme json，返回 PlantLabels（原图坐标）。

    image_size: (w, h) 磁盘上原图的实际尺寸；给了就校验与 json 里记录的一致
                （标注画在别的尺寸上会导致掩码整体错位，必须报错而不是静默继续）。
    verbose: 打印异常/未知 label 的告警（每张图每类只打一次）。
    """
    json_path = Path(json_path)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    data.pop("imageData", None)          # 大文件的坑：解析后立刻丢弃

    lab = PlantLabels(path=json_path)
    warns = []

    if image_size is not None:
        w, h = int(image_size[0]), int(image_size[1])
        jw, jh = data.get("imageWidth"), data.get("imageHeight")
        if jw is not None and jh is not None and (int(jw), int(jh)) != (w, h):
            raise ValueError(
                f"{json_path.name}: 标注尺寸 {int(jw)}x{int(jh)} 与图片实际尺寸 "
                f"{w}x{h} 不一致，掩码会整体错位。请重新导出后再标注。")

    shapes = data.get("shapes") or []
    labels_seen = []
    for s in shapes:
        label = (s.get("label") or "").strip()
        labels_seen.append(label)
        pts = _finite_points(s.get("points"))
        if len(pts) != len(s.get("points") or []):
            warns.append(f"有 {len(s.get('points') or []) - len(pts)} 个非有限坐标点被丢弃")

        if label == SETT_LABEL:
            if len(pts) >= 3:
                lab.setts.append(pts)
            else:
                warns.append(f"{SETT_LABEL} 只有 {len(pts)} 个点，忽略")
        elif label == ABOVE_LABEL:
            if len(pts) < 2:
                warns.append(f"{ABOVE_LABEL} 只有 {len(pts)} 个点，忽略")
                continue
            if len(pts) == 2:
                # 2 点 = 轴对齐矩形 → 展开成 4 个角（**这种画法有倾角偏差**，见模块 docstring）
                (xa, ya), (xb, yb) = pts
                poly = [(min(xa, xb), min(ya, yb)), (max(xa, xb), min(ya, yb)),
                        (max(xa, xb), max(ya, yb)), (min(xa, xb), max(ya, yb))]
                warns.append(f"{ABOVE_LABEL} 是 2 点轴对齐矩形 → 株高会随茎倾角偏大"
                             f"（实测最多 +19%），建议改成沿茎轴的 4 点矩形")
            else:
                # 4 点及以上：**原样保留**。压成外接框会把旋转信息丢掉，偏差就回来了。
                poly = pts
            lab.above_polys.append(poly)
        else:
            # 未知 label 一律吼出来 —— root_model 那边这里是静默忽略，
            # 结果 above_ground 被悄悄丢掉、模型拿不到标签还不报错。
            warns.append(f"未知 label {label!r}，已忽略（是不是该加进 labelme.py 的常量？）")

    lab.info = {
        "n_shapes": len(shapes),
        "labels": labels_seen,
        "n_sett": len(lab.setts),
        "above_rect": lab.above_rect,
        "warns": warns,
    }
    if image_size is not None:
        w, h = int(image_size[0]), int(image_size[1])
        if lab.setts:
            area = sum(_polygon_area(p) for p in lab.setts)
            lab.info["sett_area_ratio"] = area / float(w * h)
        if lab.above_rect:
            x0, y0, x1, y1 = lab.above_rect
            lab.info["above_area_ratio"] = (
                max(0.0, x1 - x0) * max(0.0, y1 - y0) / float(w * h))

    if verbose and warns:
        print(f"[标注告警] {json_path.name}: " + "；".join(sorted(set(warns))))
    return lab
