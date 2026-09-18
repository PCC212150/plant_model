"""从三张掩码算株高 —— 本项目的核心几何，口径见 config.py 顶部那段说明。

    株高 = 地上部（above）沿「茎轴方向」的最大伸出量，从基部起算。
    茎轴 = shoot 掩码的主轴（PCA）；基部 = shoot 两端里离种茎（sett）更近的那端。

为什么这么绕、不直接量外接框：
    植株是平铺拍的，绕茎轴滚一个角度就会让叶片在**垂直于茎轴**的方向上被压掉；
    但沿茎轴方向的分量是**刚体旋转不变量**。所以只要拿准茎轴，量出来的数就不受摆法影响
    （实测：茎翘 31.4° 时，外接框 x 跨度比沿茎轴量法大 22.5%）。

这个模块与 tool/measure_plant/measure_plant.py **必须给出同一个数**：
那边从**标注**（json 矩形 + rsml 折线）算，这边从**预测掩码**算。
验证方式就是把标注画成的掩码喂进这里，看是否复现那边的结果（见 readme「验证 A」）。

坐标约定：掩码是 (h, w) 的 bool，对外一律用 (x, y)，与项目其它部分一致。
"""
import numpy as np

# 测量有效性的判据（不满足就 measure_ok=False，并写明原因）
MIN_SHOOT_PX = 50          # shoot 像素太少 → 主轴不可信
MIN_ABOVE_PX = 500         # above 像素太少 → 株高没有意义
HEIGHT_VS_SHOOT_MIN = 0.90  # 株高 / 茎长 的下限：
                            # above 区域包含 shoot，所以沿茎轴的最大投影**必然 ≥ 茎长**。
                            # 低于这个比值说明两个通道互相矛盾（模型对同一块地方给出了
                            # 不一致的预测），结果不可信。


def _pca_axis(xy: np.ndarray):
    """返回 (质心, 主轴单位向量)。xy: (N, 2) float，列是 (x, y)。"""
    c = xy.mean(axis=0)
    d = xy - c
    cov = d.T @ d / max(len(d), 1)
    vals, vecs = np.linalg.eigh(cov)          # eigh 返回升序
    u = vecs[:, int(np.argmax(vals))]
    return c, u


def _xy(mask: np.ndarray) -> np.ndarray:
    """bool 掩码 → (N, 2) 的 (x, y) 数组。"""
    ys, xs = np.nonzero(mask)
    return np.stack([xs, ys], axis=1).astype(np.float64)


def measure_height(shoot: np.ndarray, above: np.ndarray, sett: np.ndarray,
                   min_shoot_px: int = MIN_SHOOT_PX,
                   min_above_px: int = MIN_ABOVE_PX) -> dict:
    """三张掩码（同尺寸 bool）→ 株高与诊断量。

    返回 dict：
        height       株高(px)；测量失败时为 0.0
        ok           测量是否有效
        reasons      无效的原因列表（ok=True 时可能是空的告警）
        base         基部坐标 (x, y)
        tip          shoot 另一端坐标 (x, y)
        axis          茎轴单位向量 (ux, uy)，指向从基部往叶鞘端
        shoot_len    茎长(px，沿主轴的跨度)
        above_px     above 的像素数
        sett_px      sett 的像素数
        base_from_sett  True=基部由 sett 判定；False=sett 缺失、退化用了"靠右那端"的兜底
    """
    out = {"height": 0.0, "ok": False, "reasons": [], "base": None, "tip": None,
           "axis": None, "shoot_len": 0.0, "above_px": int(above.sum()),
           "sett_px": int(sett.sum()), "base_from_sett": False}

    s_xy = _xy(shoot)
    if len(s_xy) < min_shoot_px:
        out["reasons"].append(f"shoot 掩码太小({len(s_xy)}px < {min_shoot_px})，主轴不可信")
        return out
    if out["above_px"] < min_above_px:
        out["reasons"].append(f"above 掩码太小({out['above_px']}px < {min_above_px})")
        return out

    _, u = _pca_axis(s_xy)
    proj_s = s_xy @ u                                  # shoot 各点在主轴上的投影
    i_lo, i_hi = int(np.argmin(proj_s)), int(np.argmax(proj_s))
    p_lo, p_hi = s_xy[i_lo], s_xy[i_hi]
    shoot_len = float(proj_s[i_hi] - proj_s[i_lo])

    # 基部 = 离种茎更近的那端
    t_xy = _xy(sett)
    if len(t_xy):
        c_sett = t_xy.mean(axis=0)
        base, tip = ((p_lo, p_hi) if np.linalg.norm(p_lo - c_sett) <= np.linalg.norm(p_hi - c_sett)
                     else (p_hi, p_lo))
        out["base_from_sett"] = True
    else:
        # sett 没预测出来时的兜底：这套设备里种茎**永远在画面右侧**（茎朝左铺），
        # 所以取 x 更大的那端当基部。这是设备先验，不是普适规律 —— 记一条告警。
        base, tip = (p_hi, p_lo) if p_hi[0] >= p_lo[0] else (p_lo, p_hi)
        out["reasons"].append("sett 掩码为空，退化用「靠右那端是基部」的设备先验")
    if np.linalg.norm(tip - base) < 1e-6:
        out["reasons"].append("shoot 两端重合，无法定向")
        return out

    u = (tip - base) / np.linalg.norm(tip - base)       # 重定向：基部 → 叶鞘端
    a_xy = _xy(above)
    proj_a = (a_xy - base) @ u
    height = float(max(proj_a.max(), 0.0))              # 折返回基部后方的点不计入高度

    out.update({"height": height, "base": tuple(base), "tip": tuple(tip),
                "axis": (float(u[0]), float(u[1])), "shoot_len": shoot_len})

    if shoot_len > 0 and height < HEIGHT_VS_SHOOT_MIN * shoot_len:
        out["reasons"].append(
            f"株高({height:.0f}) < {HEIGHT_VS_SHOOT_MIN:g}×茎长({shoot_len:.0f})，"
            f"两通道互相矛盾")
        return out

    out["ok"] = True
    return out


def heights_of_masks(masks: dict, **kw) -> dict:
    """方便调用：masks 是 {"shoot": bool, "above": bool, "sett": bool}。"""
    return measure_height(masks["shoot"], masks["above"], masks["sett"], **kw)
