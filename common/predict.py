"""单图推理 + 株高测量：原图 → 三通道掩码 → 株高。

与 root_model/common/predict.py 的差别（**不要照抄那边的后处理**）：
    那边是为一堆细根设计的，用了滞回低阈值（把断成几截的细根接回来）和
    「检查范围」ROI（把统计限制在托盘内）。本项目的三个目标都不是细线断续问题：
    `above` 是一大块区域、`sett` 是个小圆斑、`shoot` 虽细但只用来定主轴方向，
    断一小段不影响主轴。所以这里 **只用固定阈值**，逻辑简单得多。

测量在**原图分辨率**上做（掩码先上采样回去），而不是在模型分辨率上量完再乘系数 ——
上采样会改变边界形状，而株高对"最远那一点"极敏感；直接在原图尺度量，
才能和 tool/measure_plant 从标注算出来的数直接对比（那边也是原图尺度）。
"""
import numpy as np
import torch

import config
from common import image_io
from common.measure import measure_height


def _forward_prob(model, x):
    """前向并取 sigmoid。model 可以是单个模型，也可以是列表（集成按概率平均）。"""
    if isinstance(model, (list, tuple)):
        if not model:
            raise ValueError("模型列表为空")
        acc = None
        for m in model:
            p = torch.sigmoid(m(x).float())
            acc = p if acc is None else acc + p
        return acc / len(model)
    return torch.sigmoid(model(x).float())


@torch.no_grad()
def predict(model, img, max_side=None, stride=None, device="cuda", threshold=None):
    """一张原图（uint8 (h0, w0, 3)）→ 概率、原图尺度掩码、株高测量结果。

    返回 dict：
        probs        (C, h1, w1) float32，CPU tensor（模型分辨率）
        masks_orig   [bool(h0, w0), ...] 三通道原图尺度掩码（overlay 与测量都用它）
        measure      common.measure.measure_height 的结果；height 是**原图尺度**像素
        target_size  (w1, h1) 模型输入尺寸
    """
    if max_side is None:
        max_side = config.MAX_SIDE
    if stride is None:
        stride = config.STRIDE
    if threshold is None:
        threshold = config.PRED_THRESHOLD

    h0, w0 = img.shape[:2]
    w1, h1 = image_io.target_size(w0, h0, max_side, stride)
    x = image_io.to_model_input(image_io.resize_rgb(img, w1, h1)).to(device)
    probs = _forward_prob(model, x)[0].cpu()                 # (C, h1, w1)

    masks_orig = [image_io.prob_to_orig_mask(probs.unsqueeze(0), w0, h0,
                                             threshold=threshold, channel=c)
                  for c in range(probs.shape[0])]
    res = measure_height(*masks_orig[:3])
    return {"probs": probs, "masks_orig": masks_orig, "measure": res,
            "target_size": (w1, h1)}


def make_overlay(img, masks_orig, measure=None, alpha=0.45):
    """原图 + 三通道着色：above 半透明填充、shoot 红、sett 橙；再画茎轴与基部。

    img 是 uint8 (h,w,3)；masks_orig 是 [shoot, above, sett] 的原图尺度 bool。
    返回 uint8 (h,w,3)。
    """
    out = img.astype(np.float32).copy()
    shoot, above, sett = masks_orig[:3]
    # above 先铺底（面积最大，半透明），再盖细的 shoot / sett，保证细结构看得见
    out[above] = out[above] * (1 - alpha) + np.array([0, 160, 255]) * alpha
    out[shoot] = out[shoot] * (1 - alpha) + np.array([255, 40, 40]) * alpha
    out[sett] = out[sett] * (1 - alpha) + np.array([255, 170, 0]) * alpha

    if measure and measure.get("ok") and measure.get("axis"):
        import cv2
        bx, by = measure["base"]
        ux, uy = measure["axis"]
        h = measure["height"]
        cv2.line(out, (int(bx), int(by)), (int(bx + ux * h), int(by + uy * h)),
                 (0, 255, 0), 8, cv2.LINE_AA)          # 绿线 = 量株高的那条线
        cv2.circle(out, (int(bx), int(by)), 24, (0, 255, 0), -1)   # 绿点 = 基部
    return np.clip(out, 0, 255).astype(np.uint8)
