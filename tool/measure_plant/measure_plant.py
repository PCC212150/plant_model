"""从 labelme json + RSML 标注里量株高与「基点到叶尖」长度，并筛查摆歪的植株。

标注口径（**2026-09-18 与用户确认**）：

    json  above_ground  矩形 = 地上部的外接框。
    json  stem          polygon = 种茎的横截面（不是新茎）。
    rsml  1.1           从「幼苗与种茎之间的基点」到「可见的叶鞘开口顶端」= 茎，
                        也是**茎轴 = 植株的自然竖直方向**。
    rsml  1.1.1, 1.1.2… 各叶片折线，起点都接在 1.1 的末端（叶鞘顶端）。
                        **基点到叶尖长度 = |1.1| + |1.1.x|**（先走茎，再走叶）。

株高怎么算（**这里是全工具最关键的一步**）：

    植株是**平铺**拍的，所以"站起来的高度"= 躺着时**沿茎轴方向**伸出多远。

    **株高(投影) = 地上部所有点在茎轴方向上的最大伸出量** ← 推荐
    株高(x跨度) = above_ground 矩形的 x 跨度                  ← 用户原口径

    为什么投影比 x 跨度好：平放时植株绕茎轴滚一个角度，叶片在**垂直于茎轴**方向的分量
    会被压掉，但**沿茎轴方向的分量是刚体旋转不变量** —— 滚多少都不影响。而 x 跨度里
    混进了茎本身倾斜的分量，**实测茎翘 31° 时 x 跨度比真值大 22.5%**。
    茎水平时两者等价（±10° 内差 2% 以内）。

两个判据（互相独立）：

    ① 最远点是不是叶尖：叶尖的投影 ÷ 地上部最大投影。远小于 1 说明最远点是叶身中段
       （叶折返/堆叠），这时"最远点"的位置会随摆放方式变 → 该图不可与其它时点比。
    ② 时点自洽：同一植株相邻时点的株高**不该明显下降**（植株是长的）。
       下降超过 --drop-tol → 可疑。

    **实测**：8 株里 6 株两个判据都过（叶尖比 1.00，时点变化 −4% ~ +5%），
    2 株被挑出（C001-2_1231 叶折返 + 降 47%；C001-3_1231 降 21%），
    与人工看图确认的"摆放角度变化很大"一致。

用法：
    python measure_plant.py --dir "..\\datasets\\plant" --dry-run     # 只看筛查报告
    python measure_plant.py --dir "..\\datasets\\plant"               # 出 CSV
    python measure_plant.py --dir "..." --mm-per-px 0.19              # 顺带出毫米
"""
import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree

LOG_NAME = "plant_measure.csv"

# 筛查阈值
PROJ_RATIO_MIN = 0.80      # ① 叶尖投影 / 株高
DROP_TOL = 0.10            # ② 时点下降容忍（10%）


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="量株高与基点到叶尖长度，并筛查摆歪的植株",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=('示例：\n'
                '  python measure_plant.py --dir "..\\datasets\\plant" --dry-run\n'
                '  python measure_plant.py --dir "..\\datasets\\plant" --mm-per-px 0.19\n'),
    )
    p.add_argument("--dir", required=True, help="含 jpg + json + rsml 的数据集文件夹")
    p.add_argument("--out", default=None, help=f"CSV 输出路径（默认 <dir>\\{LOG_NAME}）")
    p.add_argument("--mm-per-px", type=float, default=0.0,
                   help="像素→毫米换算（0 = 只输出像素）。刻度板一格 26px，一格多少毫米填进来即可")
    p.add_argument("--drop-tol", type=float, default=DROP_TOL,
                   help=f"同一植株相邻时点的株高下降容忍，超过即标可疑（默认 {DROP_TOL:g}）")
    p.add_argument("--dry-run", action="store_true", help="只打印筛查报告，不写 CSV")
    return p.parse_args(argv)


def plant_key(name: str) -> str:
    """'plant_C001-1_20241229CK' -> 'C001-1'（同株的多个时点归一组）。"""
    s = str(name).replace(" ", "")
    for pre in ("plant_", "root_"):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    head, sep, tail = s.rpartition("_")
    if sep and len(tail) >= 6 and any(c.isdigit() for c in tail):
        s = head
    return s


def _collect_points(el):
    """收集某个 <root> **直属**几何里的点，**跳过嵌套的子 <root>**。

    RSML 的根是嵌套写的：`1.1.1` 整个写在 `1.1` 内部。所以必须显式跳过
    `child.tag == "root"`，否则父根会把所有子根的点都算进自己的长度里
    —— 实测会把 963px 的茎算成 2809px。子根由 walk() 单独收，不会漏。
    """
    pts = []
    for child in el:
        if child.tag == "root":
            continue
        if child.tag == "point":
            x, y = child.get("x"), child.get("y")
            if x is not None and y is not None:
                pts.append((float(x), float(y)))
        else:
            pts.extend(_collect_points(child))
    return pts


def parse_rsml(path: Path):
    """返回 {ID: [(x, y), ...]}，含任意嵌套层级的根（文档序）；少于 2 点的忽略。

    与 root_model 的 common/rsml_parse.py 语义一致（那边是权威实现）。
    """
    tree = ElementTree.parse(path)
    xml_root = tree.getroot()
    out = {}

    def walk(el):
        for child in el:
            if child.tag == "root":
                pts = _collect_points(child)
                if len(pts) >= 2:
                    out[child.get("ID", "")] = pts
            walk(child)

    scene = xml_root.find("scene")
    walk(scene if scene is not None else xml_root)
    return out


def polyline_len(pts) -> float:
    return sum(((pts[i+1][0]-pts[i][0])**2 + (pts[i+1][1]-pts[i][1])**2) ** 0.5
               for i in range(len(pts)-1))


def above_ground_box(json_path: Path):
    d = json.loads(json_path.read_text(encoding="utf-8"))
    for s in d.get("shapes", []):
        if s.get("label") == "above_ground":
            xs = [p[0] for p in s["points"]]
            ys = [p[1] for p in s["points"]]
            return min(xs), min(ys), max(xs), max(ys)
    return None


def measure_one(stem: str, folder: Path):
    """量一株，返回 dict 或 None（缺文件/缺标注）。"""
    jp, rp = folder / f"{stem}.json", folder / f"{stem}.rsml"
    if not (jp.exists() and rp.exists()):
        return None
    box = above_ground_box(jp)
    roots = parse_rsml(rp)
    if box is None or "1.1" not in roots:
        return None

    x0, y0, x1, y1 = box
    span = x1 - x0                     # 用户原口径：矩形 x 跨度
    aspect = span / (y1 - y0) if y1 > y0 else float("nan")

    base = roots["1.1"][0]
    sheath = roots["1.1"][-1]
    stem_len = polyline_len(roots["1.1"])

    leaves = {k: v for k, v in roots.items() if k.startswith("1.1.") and k != "1.1"}
    # 基点到叶尖 = 先走茎(1.1)，再走这条叶(1.1.x)
    leaf_total = {k: stem_len + polyline_len(v) for k, v in leaves.items()}

    # 茎轴单位向量（基部 → 叶鞘顶端）。自然竖直方向就是它。
    ux, uy = sheath[0] - base[0], sheath[1] - base[1]
    n = (ux*ux + uy*uy) ** 0.5 or 1.0
    ux, uy = ux / n, uy / n
    tilt = math.degrees(math.atan2(-(uy), -(ux)))     # 相对水平（向上为正）

    def proj_of(p):
        return (p[0] - base[0]) * ux + (p[1] - base[1]) * uy

    # 校正株高 = 地上部**所有点**沿茎轴的最大伸出量。
    # 为什么不是 x 跨度：平放时植株绕茎轴滚一个角度，叶片在垂直于茎轴方向的分量会被
    # 压掉，但**沿茎轴方向的分量是刚体旋转不变量**；而 x 跨度里混进了茎本身倾斜的分量
    # （实测茎翘 31° 时 x 跨度比真值大 22.5%）。
    all_pts = [p for v in leaves.values() for p in v] + roots["1.1"]
    height_proj = max(proj_of(p) for p in all_pts)

    # 判据①：最远那条叶的**叶尖**是否就是最远点。远小于 1 说明最远点是叶身中段
    # （叶折返/堆叠），这时"最远点"的位置会随摆放方式变化 → 可疑。
    tip_proj = max((proj_of(v[-1]) for v in leaves.values()), default=0.0)
    tip_ratio = tip_proj / height_proj if height_proj > 0 else 0.0

    return {
        "植株": stem, "归属": plant_key(stem),
        "株高_投影px": height_proj, "株高_x跨度px": span,
        "茎倾角deg": tilt, "茎长px": stem_len,
        "叶数": len(leaves),
        "最长_基点到叶尖px": max(leaf_total.values()) if leaf_total else 0.0,
        "长宽比": aspect, "叶尖投影比": tip_ratio,
        "_leaf_totals": [leaf_total[k] for k in sorted(leaves)],
    }


def main(argv=None):
    args = parse_args(argv)
    folder = Path(args.dir)
    if not folder.is_dir():
        sys.exit(f"[错误] 文件夹不存在: {folder}")

    stems = sorted({p.stem for p in folder.glob("*.jpg")} |
                   {p.stem for p in folder.glob("*.png")})
    if not stems:
        sys.exit(f"[错误] {folder} 里没有图片")

    rows, skipped = [], []
    for s in stems:
        r = measure_one(s, folder)
        (rows.append(r) if r else skipped.append(s))
    if not rows:
        sys.exit("[错误] 没有任何一株同时具备 json(above_ground) 与 rsml(1.1)")
    for s in skipped:
        print(f"[跳过] {s}：缺 json/rsml，或缺 above_ground / 1.1")

    # ---------- 判据①几何自洽 ----------
    for r in rows:
        reasons = []
        if r["叶尖投影比"] < PROJ_RATIO_MIN:
            reasons.append(f"最远点不是叶尖(叶尖比{r['叶尖投影比']:.2f}<{PROJ_RATIO_MIN:g})")
        r["_reasons"] = reasons

    # ---------- 判据②时点自洽 ----------
    by_plant = defaultdict(list)
    for r in rows:
        m = re.search(r"_(\d{8})", r["植株"])
        if m:
            by_plant[r["归属"]].append((m.group(1), r))

    for key, items in sorted(by_plant.items()):
        items.sort()
        for (d_prev, prev), (d_cur, cur) in zip(items, items[1:]):
            if prev["株高_投影px"] <= 0:
                continue
            change = cur["株高_投影px"] / prev["株高_投影px"] - 1
            cur[f"_变化vs{d_prev}"] = change
            if change < -args.drop_tol:
                cur["_reasons"].append(
                    f"比{d_prev}降了{-change*100:.0f}%（{args.drop_tol*100:g}%容忍）")

    # ---------- 报告 ----------
    print(f"\n{'植株':26} {'株高_投影':>9} {'x跨度':>6} {'茎倾角':>7} {'茎长':>6} "
          f"{'叶':>3} {'最长(基→尖)':>11} {'叶尖比':>7} {'判定':>6}")
    for r in rows:
        verdict = "可疑" if r["_reasons"] else "OK"
        print(f"{r['植株']:26} {r['株高_投影px']:9.0f} {r['株高_x跨度px']:6.0f} "
              f"{r['茎倾角deg']:6.1f}° {r['茎长px']:6.0f} {r['叶数']:3d} "
              f"{r['最长_基点到叶尖px']:11.0f} {r['叶尖投影比']:7.2f} {verdict:>6}"
              + (f"   ← {'；'.join(r['_reasons'])}" if r["_reasons"] else ""))

    bad = [r for r in rows if r["_reasons"]]
    print(f"\n共 {len(rows)} 株：可信 {len(rows)-len(bad)}，可疑 {len(bad)}"
          + (f"（{'、'.join(r['植株'] for r in bad)}）" if bad else ""))
    if bad:
        print("可疑株的株高不可直接使用 —— 多半是拍摄时植株没沿茎轴向铺平（见文件头说明）。")

    if args.dry_run:
        print("\n（--dry-run：没写 CSV）")
        return 0

    out = Path(args.out) if args.out else folder / LOG_NAME
    mm = args.mm_per_px

    def leaf_list(r, scale=1.0):
        return ";".join(f"{v*scale:.0f}" if scale == 1 else f"{v*scale:.1f}"
                        for v in r["_leaf_totals"])

    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write(f"# 株高与基点到叶尖长度   数据: {folder}\n")
        fh.write("# 株高(投影)  = 地上部所有点沿「茎轴方向」的最大伸出量（推荐口径）\n")
        fh.write("#               茎轴 = rsml 1.1（基部→叶鞘顶端），自然竖直方向就是它。\n")
        fh.write("#               平放时绕茎轴滚动不改变这个量（刚体旋转不变量），而 x 跨度会\n")
        fh.write("#               混进茎本身倾斜的分量 —— 实测茎翘 31° 时 x 跨度偏大 22.5%。\n")
        fh.write("# 株高(x跨度) = above_ground 矩形 x 跨度（用户原口径，茎水平时才等于上面那个）\n")
        fh.write("# 基点到叶尖 = |rsml 1.1| + |rsml 1.1.x|（先走茎，再走这条叶）\n")
        fh.write(f"# 换算：mm_per_px = {mm:g}" + ("（只输出像素）" if not mm else "") + "\n")
        fh.write(f"# 判定：叶尖投影比 < {PROJ_RATIO_MIN:g}（最远点不是叶尖，说明叶折返/堆叠）\n")
        fh.write(f"#       或相邻时点株高(投影) 下降 > {args.drop_tol*100:g}%  → 记「可疑」\n")
        w = csv.writer(fh)

        # 表头与值成对构建，避免插入列时错位
        def build(r):
            pairs = [("图片名", r["植株"])]
            if mm:
                pairs.append(("株高(投影,mm)", f"{r['株高_投影px']*mm:.1f}"))
            pairs += [("株高(投影,px)", f"{r['株高_投影px']:.0f}"),
                      ("株高(x跨度,px)", f"{r['株高_x跨度px']:.0f}"),
                      ("茎倾角(deg)", f"{r['茎倾角deg']:.1f}"),
                      ("茎长(px)", f"{r['茎长px']:.0f}")]
            if mm:
                pairs.append(("茎长(mm)", f"{r['茎长px']*mm:.1f}"))
            pairs += [("叶数", r["叶数"]),
                      ("各叶_基点到叶尖(px)", leaf_list(r)),
                      ("最长_基点到叶尖(px)", f"{r['最长_基点到叶尖px']:.0f}")]
            if mm:
                pairs += [("各叶_基点到叶尖(mm)", leaf_list(r, mm)),
                          ("最长_基点到叶尖(mm)", f"{r['最长_基点到叶尖px']*mm:.1f}")]
            pairs += [("长宽比", f"{r['长宽比']:.2f}"),
                      ("叶尖投影比", f"{r['叶尖投影比']:.2f}"),
                      ("判定", "可疑" if r["_reasons"] else "OK"),
                      ("原因", "；".join(r["_reasons"]))]
            return pairs

        w.writerow([k for k, _ in build(rows[0])])
        for r in rows:
            w.writerow([v for _, v in build(r)])
    print(f"\nCSV: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
