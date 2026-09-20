"""批量推理：图片文件夹 → 每张一条株高记录 + 一张 overlay，最后出 CSV。

    图片文件夹里放 .jpg/.png，逐张预测并量株高。
    每张图输出  {名}_overlay.jpg  （地上部半透明蓝 + 茎红 + 种茎橙 + 绿线=量株高的那条）
    整个批次输出 result/{文件夹名}/{文件夹名}.csv

**CSV 里有一列「时点变化」**：同一植株（编号-重复）按日期排序后，与上一个时点比株高变了
多少 —— 植株是长的，**明显下降就说明那天摆得不一样**（叶片折返/堆叠会改变"最远点"落在哪）。
这是采集端的问题，模型修不了，只能标出来让人复核。阈值 --drop-tol（默认 10%）。

用法：
    python inference.py --dir "D:\\待测图片"
    python inference.py --model model_202609181504 --dir "..." --size 512
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import ckpt, image_io, naming  # noqa: E402
from common.dataset import CH_ABOVE, plant_key  # noqa: E402
from common.predict import make_overlay, predict  # noqa: E402

import re  # noqa: E402
_DATE = re.compile(r"_(\d{8})")

# 跨批次的「可疑记录」累计文件（每跑一次 inference 追加几行，方便回看哪批数据有问题）
RECORD_NAME = "可疑记录.csv"


def parse_args():
    p = argparse.ArgumentParser(description="批量预测甘蔗株高，输出 overlay 与 CSV")
    p.add_argument("--dir", required=True, help="待预测的图片文件夹")
    p.add_argument("--model", default=None,
                   help="模型文件夹名（可省 model_ 前缀；逗号分隔=集成）。留空=取最新")
    p.add_argument("--size", type=int, default=None,
                   help="输入长边。默认用权重里记录的训练 size（尺度必须与训练一致）")
    p.add_argument("--out-dir", type=Path, default=config.RESULT_DIR)
    p.add_argument("--drop-tol", type=float, default=0.10,
                   help="同一植株相邻时点株高下降超过此比例就标记（默认 0.10）")
    p.add_argument("--no-overlay", action="store_true", help="不写 overlay 图（只出 CSV）")
    p.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return p.parse_args()


def _geom(m, above_mask):
    """两个**推理时可得**的几何诊断量（不需要任何标注）。

    tilt  茎轴相对水平的夹角(°)，0~90。readme 的实测表里株高偏差随倾角单调增大。
    splay 地上部「垂直于茎轴的跨度 ÷ 沿茎轴的跨度」。铺得直时叶片沿轴伸展，这个比值小；
          叶折返/侧向摊开时变大。

    **注意：这两个量只是诊断参考，不是判据。** 拿 20 组有标注的数据核对过：
    用 rsml 折线算 splay，可疑组中位 0.76、正常组中位 0.47，但两组区间重叠
    （正常 0.24~0.99，可疑 0.43~1.05），**卡不出一个能用的阈值**。
    真正能判「歪」的是 measure_plant 的叶尖比（需要 rsml），推理时没有。
    所以这里只记录数值 + 交给下面的时点自洽性去标记。
    """
    u = m.get("axis")
    if not u or m.get("base") is None or above_mask is None or not above_mask.any():
        return None, None
    u = np.asarray(u, float)
    tilt = abs(np.degrees(np.arctan2(u[1], u[0])))
    tilt = min(tilt, 180.0 - tilt)
    ys, xs = np.nonzero(above_mask)
    rel = np.column_stack([xs, ys]).astype(np.float64) - np.asarray(m["base"], float)
    pu, pv = rel @ u, rel @ np.array([-u[1], u[0]])
    span = pu.max() - pu.min()
    return (float(tilt), float((pv.max() - pv.min()) / span) if span > 1e-6 else None)


def main():
    args = parse_args()
    img_dir = Path(args.dir)
    if not img_dir.is_dir():
        raise SystemExit(f"[错误] 图片文件夹不存在: {img_dir}")

    # 同一文件夹里不允许有同名不同扩展名（会互相覆盖输出）
    stems = {}
    for p in sorted(img_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTS:
            if p.stem in stems:
                raise SystemExit(f"[错误] {p.stem} 有多个扩展名（{stems[p.stem].name} / "
                                 f"{p.name}），输出会互相覆盖。请先清理。")
            stems[p.stem] = p
    if not stems:
        raise SystemExit(f"[错误] {img_dir} 里没有图片"
                         f"（支持的扩展名: {sorted(config.IMAGE_EXTS)}）")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    pths, names = ckpt.resolve_pths(args.model)
    models, metas = ckpt.load_models(pths, device)
    size = args.size or metas[0].get("size") or config.MAX_SIDE
    print(f"模型: {', '.join(names)} | 输入长边 {size} | 设备 {device}")
    print(f"待预测: {len(stems)} 张（{img_dir}）")

    out_dir = naming.create_unique_dir(args.out_dir, img_dir.name)
    print(f"输出目录: {out_dir}\n")

    rows = []
    t0 = time.time()
    for k, (stem, path) in enumerate(sorted(stems.items()), 1):
        img = image_io.load_rgb(path)
        res = predict(models, img, max_side=size, device=device)
        m = res["measure"]
        if not args.no_overlay:
            ov = make_overlay(img, res["masks_orig"], m)
            Image.fromarray(ov).save(out_dir / f"{stem}_overlay.jpg",
                                     quality=88, subsampling=0)
        dt = _DATE.search(stem)
        tilt, splay = _geom(m, res["masks_orig"][CH_ABOVE]
                            if res["masks_orig"][CH_ABOVE] is not None else None)
        rows.append({
            "stem": stem, "归属": plant_key(stem),
            "date": dt.group(1) if dt else "",
            "height": m["height"], "ok": m["ok"], "reasons": "；".join(m["reasons"]),
            "shoot_len": m["shoot_len"], "above_px": m["above_px"],
            "sett_px": m["sett_px"], "base_from_sett": m["base_from_sett"],
            "tilt": tilt, "splay": splay,
        })
        print(f"  [{k}/{len(stems)}] {stem:30} 株高 {m['height']:7.0f}px"
              + ("" if m["ok"] else f"   [无效] {rows[-1]['reasons']}"))

    # ---- 时点变化筛查：同植株按日期排序，株高不该明显下降 ----
    by_plant = defaultdict(list)
    for r in rows:
        if r["date"]:
            by_plant[r["归属"]].append(r)
    for _, items in by_plant.items():
        items.sort(key=lambda r: r["date"])
        for prev, cur in zip(items, items[1:]):
            if prev["height"] > 0 and prev["ok"] and cur["ok"]:
                ch = cur["height"] / prev["height"] - 1
                cur["变化"] = f"{ch * 100:+.1f}%"
                if ch < -args.drop_tol:
                    cur["reasons"] = (cur["reasons"] + "；" if cur["reasons"] else "") + \
                        f"比 {prev['date']} 降了 {-ch * 100:.0f}%（{args.drop_tol * 100:g}% 容忍）"

    bad = [r for r in rows if not r["ok"] or "降了" in r["reasons"]]
    print(f"\n共 {len(rows)} 张：正常 {len(rows) - len(bad)}，可疑 {len(bad)}；"
          f"耗时 {time.time() - t0:.1f}s")

    # ---- 可疑清单：单独成文件 + 跨批次累计，方便回看哪批数据有问题 ----
    # 判据只有两条，**都是推理时可得的**：
    #   ① measure 的通道一致性守卫没过（ok=False）——above 与 shoot 互相矛盾
    #   ② 时点自洽：同植株相邻时点株高下降超过 --drop-tol
    # measure_plant 还有一条「叶尖比」（最远点是不是叶尖），那个要 rsml，推理时算不了 ——
    # 实测 7 个可疑样本里有 4 个只有它报得出来，所以这份清单只能覆盖一部分。
    if bad:
        print(f"\n=== 可疑清单（{len(bad)} 张，建议人工看一眼 overlay）===")
        for r in bad:
            extra = f"  倾角{r['tilt']:.0f}° 垂直/沿轴{r['splay']:.2f}" \
                if r["tilt"] is not None and r["splay"] is not None else ""
            print(f"  {r['stem']:30} 株高 {r['height']:7.0f}px{extra}")
            print(f"      {r['reasons'] or '（无备注）'}")
        lst = out_dir / "可疑清单.txt"
        with open(lst, "w", encoding="utf-8") as fh:
            fh.write(f"# 可疑清单   批次: {img_dir}\n")
            fh.write("# 判据：① 通道一致性守卫没过  ② 同植株相邻时点株高下降超过 "
                     f"{args.drop_tol*100:g}%\n")
            fh.write("# 注意：measure_plant 那条「叶尖比」需要 rsml，推理时算不了，"
                     "所以这份清单覆盖不全\n\n")
            for r in bad:
                fh.write(f"{r['stem']}\t株高{r['height']:.0f}px\t{r['reasons'] or ''}\n")
        print(f"可疑清单: {lst}")

        rec = config.RESULT_DIR / RECORD_NAME
        rec.parent.mkdir(parents=True, exist_ok=True)
        new = not rec.exists()
        with open(rec, "a", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["批次", "时间", "图片名", "归属", "日期", "株高(px)",
                            "茎长(px)", "茎倾角(°)", "垂直/沿轴", "原因"])
            ts = time.strftime("%Y-%m-%d %H:%M")
            for r in bad:
                w.writerow([img_dir.name, ts, r["stem"], r["归属"], r["date"],
                            f"{r['height']:.0f}", f"{r['shoot_len']:.0f}",
                            "" if r["tilt"] is None else f"{r['tilt']:.1f}",
                            "" if r["splay"] is None else f"{r['splay']:.2f}",
                            r["reasons"]])
        print(f"已追加到累计记录: {rec}")

    # ---- CSV ----
    out = out_dir / f"{out_dir.name}.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write(f"# 株高预测   模型: {', '.join(names)}   输入长边: {size}\n")
        fh.write("# 株高 = 地上部沿「茎轴方向」的最大伸出量，从基部(种茎侧)起算；单位=像素\n")
        fh.write("# 采集端没有已知尺寸的参照物，所以没有毫米列（config.MM_PER_PX=0）\n")
        fh.write(f"# 「时点变化」= 同一植株与上一个拍摄日相比的株高变化。"
                 f"明显下降(>{args.drop_tol*100:g}%)说明那天摆放方式变了 —— "
                 f"叶片折返会改变最远点落在哪，模型修不了，需要复核\n")
        fh.write("# 「茎倾角」= 茎轴相对水平的夹角；「垂直/沿轴」= 地上部垂直于茎轴的跨度÷沿茎轴\n"
                 "#   的跨度。两列只是诊断参考、不是判据 —— 实测它们分不开正常与可疑\n"
                 "#   （详见 inference.py 的 _geom 注释）\n")
        w = csv.writer(fh)
        w.writerow(["图片名", "株高(px)", "茎长(px)", "地上部面积(px²)", "种茎面积(px²)",
                    "基部来自种茎", "测量有效", "时点变化", "茎倾角(°)", "垂直/沿轴", "备注"])
        for r in rows:
            w.writerow([r["stem"], f"{r['height']:.0f}", f"{r['shoot_len']:.0f}",
                        r["above_px"], r["sett_px"],
                        "是" if r["base_from_sett"] else "否",
                        "是" if r["ok"] else "否",
                        r.get("变化", ""),
                        "" if r["tilt"] is None else f"{r['tilt']:.1f}",
                        "" if r["splay"] is None else f"{r['splay']:.2f}",
                        r["reasons"]])
    print(f"CSV: {out}")
    if not args.no_overlay:
        print(f"overlay: {out_dir}\\*_overlay.jpg")


if __name__ == "__main__":
    main()
