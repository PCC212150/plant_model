"""二值分割指标：IoU / Dice / 像素准确率（在预测掩码与 GT 掩码间计算）。"""
import numpy as np


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """pred/gt: 同形状 bool 掩码。返回 {'iou','dice','accuracy'}。"""
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(pred.size - tp - fp - fn)
    denom = tp + fp + fn
    iou = tp / denom if denom > 0 else 0.0
    dice = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    acc = (tp + tn) / float(pred.size) if pred.size else 0.0
    return {"iou": iou, "dice": dice, "accuracy": acc}


def multi_channel_metrics(preds, gts, names=None, valid=None) -> list:
    """逐通道算指标：preds/gts 为同长度的掩码列表（形状一致）。

    valid: 可选的逐通道 0/1 列表；为 0 的通道返回 {'iou': nan, ...}（该通道没有真值）。
    返回 [{'name','iou','dice','accuracy'}, ...]，顺序与输入一致。
    """
    out = []
    for i, (pred, gt) in enumerate(zip(preds, gts)):
        name = names[i] if names else str(i)
        if valid is not None and not valid[i]:
            out.append({"name": name, "iou": float("nan"),
                        "dice": float("nan"), "accuracy": float("nan")})
            continue
        m = binary_metrics(pred, gt)
        m["name"] = name
        out.append(m)
    return out


def nanmean(values) -> float:
    """忽略 nan 求均值；全为 nan 时返回 nan。"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    with np.errstate(invalid="ignore"):
        return float(np.nanmean(arr))
