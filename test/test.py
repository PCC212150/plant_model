"""评测：逐图对比「标注算的株高」与「模型预测的株高」，并给出三通道 Dice/IoU。

**核心指标是株高误差，不是 Dice。** Dice 只说明分割画得像不像，而株高取决于
「最远那一点落在哪」—— 分割差一点点，最远点就可能差很多。这两个数要分开看：

    GT 株高   从标注（json 旋转矩形 + rsml 折线）算，是基准
    预测株高  从模型输出的三通道掩码算（common/measure.py）
    误差      两者之差。它同时包含「分割误差」和「测量链路误差」

想分清这两者：把标注画成的 GT 掩码也喂进 measure.py 量一遍（见 --gt-masks），
得到的误差就是**测量链路自身**的误差，与模型无关。

用法：
    python test/test.py --model model_202609181504
    python test/test.py --model model_202609181504 --size 512 --data-dir datasets/plant
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import ckpt, image_io, naming  # noqa: E402
from common.dataset import (CH_ABOVE, CH_SETT, CH_SHOOT, build_target_masks,  # noqa: E402
                            discover_pairs)
from common.measure import measure_height  # noqa: E402
from common.predict import predict  # noqa: E402

N_CH = len(config.CLASS_NAMES)


def parse_args():
    p = argparse.ArgumentParser(description="评测株高模型：GT 株高 vs 预测株高 + 三通道 Dice")
    p.add_argument("--model", default=None,
                   help="模型文件夹名（可省 model_ 前缀；逗号分隔=集成）。留空=取最新")
    p.add_argument("--size", type=int, default=None,
                   help="输入长边。默认用权重里记录的训练 size（尺度必须与训练一致）")
    p.add_argument("--data-dir", type=Path, default=config.PLANT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=config.MODEL_DIR)
    p.add_argument("--mask-width", type=float, default=config.MASK_LINE_WIDTH)
    p.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    pths, names = ckpt.resolve_pths(args.model, args.out_dir)
    models, metas = ckpt.load_models(pths, device)
    size = args.size or metas[0].get("size") or config.MAX_SIDE
    print(f"模型: {', '.join(names)} | 输入长边 {size} | 设备 {device}")

    pairs = discover_pairs(args.data_dir)
    if not pairs:
        raise SystemExit(f"[错误] {args.data_dir} 里没有「图片 + 同名 .rsml」配对")
    print(f"数据: {args.data_dir}（{len(pairs)} 组）\n")

    rows = []
    t0 = time.time()
    for stem, img_path, rsml_path, json_path in pairs:
        img = image_io.load_rgb(img_path)
        h0, w0 = img.shape[:2]

        # ---- 基准：从标注算株高，以及把标注画成掩码再量一遍（测量链路自证）----
        gt_masks, gt_valid = build_target_masks(rsml_path, json_path, (w0, h0),
                                                (w0, h0), args.mask_width)
        gt_m = measure_height(gt_masks[:, :, CH_SHOOT], gt_masks[:, :, CH_ABOVE],
                              gt_masks[:, :, CH_SETT])

        # ---- 模型预测 ----
        res = predict(models, img, max_side=size, device=device)
        pred_m = res["measure"]
        pred_masks = res["masks_orig"]

        # ---- 分割指标（原图尺度上算；GT 线宽与训练一致）----
        dices = []
        for c in range(N_CH):
            if gt_valid[c] < 0.5:
                dices.append(float("nan"))
                continue
            pb, gt = pred_masks[c], gt_masks[:, :, c]
            tp = float((pb & gt).sum())
            dices.append(2.0 * tp / (pb.sum() + gt.sum() + 1e-8))

        rows.append({
            "stem": stem,
            "gt_h": gt_m["height"], "gt_ok": gt_m["ok"],
            "pred_h": pred_m["height"], "pred_ok": pred_m["ok"],
            "dices": dices, "reasons": "；".join(pred_m["reasons"]),
        })
        print(f"  {stem:26} GT {gt_m['height']:7.0f}  预测 {pred_m['height']:7.0f}  "
              f"误差 {pred_m['height'] - gt_m['height']:+8.0f}px  "
              f"Dice " + "/".join("  nan" if np.isnan(d) else f"{d:.3f}" for d in dices)
              + (f"   [{rows[-1]['reasons']}]" if rows[-1]["reasons"] else ""))

    # ---- 汇总 ----
    ok_rows = [r for r in rows if r["gt_ok"] and r["pred_ok"]]
    errs = np.array([r["pred_h"] - r["gt_h"] for r in ok_rows]) if ok_rows else np.array([])
    rel = (errs / np.array([r["gt_h"] for r in ok_rows])) if ok_rows else np.array([])
    mdice = [np.nanmean([r["dices"][c] for r in rows]) for c in range(N_CH)]

    print(f"\n=== 汇总（{len(rows)} 张；其中 {len(ok_rows)} 张双方测量都有效）===")
    if len(errs):
        print(f"  株高误差: 平均 {errs.mean():+.0f}px | 绝对平均 {np.abs(errs).mean():.0f}px"
              f" | 相对 {np.abs(rel).mean()*100:.1f}% | 最大 {np.abs(rel).max()*100:.1f}%")
    else:
        print("  没有任何一张双方测量都有效 —— 检查上面的原因列")
    print("  分割 Dice: " + " | ".join(
        f"{n}={d:.3f}" for n, d in zip(config.CLASS_NAMES, mdice)))
    print(f"  耗时 {time.time() - t0:.1f}s")

    # ---- CSV ----
    out_dir = pths[0].parent
    out = naming.create_unique_file(out_dir, f"model_test_{naming.timestamp()}.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write(f"# 株高评测  模型: {', '.join(names)}  数据: {args.data_dir}  输入长边: {size}\n")
        fh.write("# GT株高 = 标注(json旋转矩形 + rsml折线)算的；预测株高 = 模型三通道掩码算的\n")
        fh.write("# 两者都由 common/measure.py 的同一套几何算出（口径见 config.py 顶部）\n")
        fh.write(f"# 判定阈值: PRED_THRESHOLD={config.PRED_THRESHOLD}；"
                 f"above 小于 {config.MEASURE_MIN_ABOVE_PX}px 视为测量失败\n")
        w = csv.writer(fh)
        w.writerow(["图片名", "GT株高(px)", "预测株高(px)", "误差(px)", "误差(%)",
                    "GT测量有效", "预测测量有效",
                    *[f"{n}_Dice" for n in config.CLASS_NAMES], "预测无效原因"])
        for r in rows:
            e = r["pred_h"] - r["gt_h"]
            w.writerow([r["stem"], f"{r['gt_h']:.0f}", f"{r['pred_h']:.0f}",
                        f"{e:+.0f}", f"{e / r['gt_h'] * 100:+.1f}%" if r["gt_h"] else "",
                        "是" if r["gt_ok"] else "否", "是" if r["pred_ok"] else "否",
                        *["" if np.isnan(d) else f"{d:.3f}" for d in r["dices"]],
                        r["reasons"]])
        if len(errs):
            w.writerow([])
            w.writerow([f"# 平均误差 {errs.mean():+.0f}px | 绝对平均 {np.abs(errs).mean():.0f}px"
                        f" | 相对 {np.abs(rel).mean()*100:.1f}%"])
            w.writerow(["# 三通道 Dice " + " | ".join(
                f"{n}={d:.3f}" for n, d in zip(config.CLASS_NAMES, mdice))])
            w.writerow(["# 注意：GT 用的是标注矩形（当前多为轴对齐，最大使株高偏大 19%）。"
                        "改成沿茎轴的 4 点矩形后这个偏差会消失。"])
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
