# -*- coding: utf-8 -*-
"""生成 7 张小论文图件的【中文审阅版】：图内文字全部中文，仅输出 PNG（300 dpi）。

绘图软件：Python 3.13 + Matplotlib（Microsoft YaHei 中文字体）。
数据源：原项目冻结结果文件（只读），本工作区成本—损失结果（只读）。
输出：本工作区 figures/fig_article_fig1..fig7_zh.png（仅 PNG，不输出 PDF）。

配色不超过三色：蓝 #1F4E79、橙 #E69F00、灰 #7A7A7A；黑/白仅作文字与底色；
系列用线型、标记与灰度区分，保证黑白打印清晰。

图号对应主稿插入位置：
  图1 研究区、林区网格与多源数据处理流程
  图2 特征旬与下一旬标签对齐、训练—校准—测试时间隔离
  图3 五折空间留出验证的 AUC 与 Top-20% 捕获率
  图4 三类模型 ROC 曲线与 PR 曲线
  图5 XGBoost 校准前后可靠性曲线
  图6 XGBoost 与简单预警基线的 Top-k 捕获曲线
  图7 成本—损失相对经济价值曲线
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             roc_auc_score, average_precision_score)

BASE = "E:/workbuddyoutput/2026-10-16-11-07-31"
PROJ = "E:/workbuddyoutput/2026-11-10-10-24-23"
OUT = PROJ + "/figures"
CL = PROJ + "/analysis_materials/cost_loss_results"
os.makedirs(OUT, exist_ok=True)

# ---------------- 统一风格 ----------------
INK    = "#1A1A1A"   # 黑（文字/基线）
SUB    = "#444444"   # 深灰（文字）
MUTE   = "#7A7A7A"   # 灰（三色之一）
GRID   = "#D8D8D8"   # 网格灰阶
BLUE   = "#1F4E79"   # 蓝（三色之一）
ORANGE = "#E69F00"   # 橙（三色之一）
GRAY   = "#7A7A7A"   # 灰（三色之一）
TINT   = "#F2F2F2"   # 中性浅灰（框填充/底色）
BG     = "#FFFFFF"   # 白（底色）

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DengXian"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.labelsize": 11.5,
    "legend.fontsize": 9.5,
    "axes.edgecolor": INK,
    "axes.linewidth": 1.1,
    "axes.labelcolor": INK,
    "xtick.color": SUB,
    "ytick.color": SUB,
    "text.color": INK,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
})


def style_axes(ax, grid=True):
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, linestyle=(0, (3, 3)), linewidth=0.9, color=GRID, alpha=0.75)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(1.1)
        ax.spines[s].set_color(INK)
    ax.tick_params(length=4, width=1.0, direction="out", pad=4)


def save(fig, stem):
    png = OUT + "/" + stem + ".png"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.10, facecolor=BG)
    plt.close(fig)
    # 仅输出 PNG；用 PIL 精确写入 dpi=(300,300)
    im = Image.open(png)
    im.save(png, dpi=(300, 300))
    print("  已输出 %s.png（300 dpi）" % stem)


def box(ax, x, y, w, h, title, sub="", fc=BG, ec=INK, fs=10.5,
        subfs=9.0, lw=1.2, rounding=0.018, tc=None, z=3):
    style = ("round,pad=0.010,rounding_size=%.3f" % rounding) if rounding else "square,pad=0.010"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                                edgecolor=ec, facecolor=fc, zorder=z))
    tc = tc or INK
    if not title and not sub:
        return
    if sub:
        ax.text(x + w / 2, y + h * 0.74, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=z + 1)
        ax.text(x + w / 2, y + h * 0.31, sub, ha="center", va="center",
                fontsize=subfs, color=SUB, zorder=z + 1, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, color=tc, zorder=z + 1, fontweight="bold")


def arrow(ax, p1, p2, color=INK, lw=1.35, ls="-", ms=13, z=2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, connectionstyle="arc3,rad=%.2f" % rad,
                                 arrowstyle="-|>", mutation_scale=ms,
                                 linewidth=lw, linestyle=ls, color=color,
                                 zorder=z, capstyle="round"))


# ==================================================================
# 图1 研究区、林区网格与多源数据处理流程
# ==================================================================
def fig1():
    gj = json.load(open(BASE + "/data/geo/gadm41_CHN_1.json", encoding="utf-8"))
    yn, others = None, []
    for f in gj["features"]:
        if f["properties"].get("NAME_1") == "Yunnan":
            yn = f["geometry"]
        else:
            others.append(f["geometry"])

    def rings(g):
        out = []
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            for i, ring in enumerate(poly):
                out.append((np.asarray(ring, dtype=float), i == 0))
        return out

    grids = pd.read_csv(BASE + "/data/processed/grids_yunnan.csv")
    static = pd.read_csv(BASE + "/data/processed/static_features.csv")
    m = grids.merge(static[["gid", "forest_frac"]], on="gid", how="left")
    forest = m.forest_frac >= 0.5

    fig = plt.figure(figsize=(13.2, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.20)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    bbox = (97.0, 106.6, 20.7, 29.6)
    xmin, xmax, ymin, ymax = bbox
    for g in others:
        for ring, _ in rings(g):
            xs, ys = ring[:, 0], ring[:, 1]
            if xs.max() < xmin or xs.min() > xmax or ys.max() < ymin or ys.min() > ymax:
                continue
            ax.fill(xs, ys, facecolor="#EFEFEF", edgecolor="none", zorder=0)
            ax.plot(xs, ys, color="#C9C9C9", lw=0.9, zorder=0.1)
    for ring, is_ext in rings(yn):
        xs, ys = ring[:, 0], ring[:, 1]
        ax.fill(xs, ys, facecolor="#FCFCFC", edgecolor="none", zorder=0.5)
        ax.plot(xs, ys, color=INK, lw=1.7 if is_ext else 0.9, zorder=8)

    ax.scatter(m.loc[~forest, "lon_c"], m.loc[~forest, "lat_c"], s=7,
               c="#BFBFBF", marker="s", linewidths=0, zorder=3, label="非林区网格（796）")
    ax.scatter(m.loc[forest, "lon_c"], m.loc[forest, "lat_c"], s=9,
               c=BLUE, marker="s", linewidths=0, alpha=0.85, zorder=4, label="林区网格（2634）")
    ax.set_xlim(97.2, 106.4); ax.set_ylim(20.9, 29.4)
    ax.set_xticks(np.arange(98, 107, 2)); ax.set_yticks(np.arange(21, 30, 2))
    ax.set_xlabel("东经 / °"); ax.set_ylabel("北纬 / °")
    ax.grid(True, linestyle=(0, (3, 3)), linewidth=0.9, color=GRID, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(1.1); s.set_color(INK)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("（a）研究区与 0.1° 林区网格掩膜", fontsize=11.5, pad=10, fontweight="bold")
    lg = ax.legend(loc="lower left", frameon=True, fancybox=False, edgecolor=INK,
                   framealpha=1.0, fontsize=9)
    lg.get_frame().set_linewidth(0.8)
    # 指北针：箭头尖端朝上为北，字母 N 置于尖端上方
    ax.text(0.975, 0.962, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11, fontweight="bold", zorder=9)
    ax.annotate("", xy=(0.975, 0.945), xytext=(0.975, 0.872),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, mutation_scale=12))
    # 比例尺（25°N 处 1° 经度约 101 km），置于地图右下空白区，避开左下图例
    deg = 200 / (111.32 * np.cos(np.deg2rad(25.0)))
    x0, y0 = 103.6, 21.45
    bk = dict(facecolor=BG, edgecolor="none", alpha=0.88, pad=1.6)
    ax.plot([x0, x0 + deg], [y0, y0], color=INK, lw=2.0, zorder=9, clip_on=False)
    for dx in (0, deg / 2, deg):
        ax.plot([x0 + dx, x0 + dx], [y0 - 0.11, y0 + 0.11], color=INK, lw=1.1, zorder=9, clip_on=False)
    ax.text(x0 + deg / 2, y0 + 0.26, "200 km", ha="center", va="bottom",
            fontsize=9, color=INK, zorder=9, bbox=bk)

    # (b) 处理流程
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
    ax2.set_title("（b）多源数据处理与评价流程", fontsize=11.5, pad=10, fontweight="bold")
    steps = [
        ("多源数据", "FIRMS 火点｜ERA5 气象｜MOD13A2 NDVI\nWorldCover｜SRTM｜WorldPop｜OSM", TINT, BLUE),
        ("0.1° 网格 × 旬对齐", "3430 个省界内网格；2634 个林区网格掩膜\n防火年度 23 个旬，前 22 旬为特征旬", TINT, BLUE),
        ("解释变量与标签", "固定 13 个解释变量\n预测下一旬是否出现林地卫星火点", TINT, ORANGE),
        ("时间划分", "2010—2018 训练｜2019—2020 校准\n2021—2023 测试（严格顺序、互不重叠）", TINT, BLUE),
        ("模型与概率校准", "XGBoost 主模型；保序回归校准器\n逻辑回归、随机森林作对照", TINT, ORANGE),
        ("评价与决策", "AUC／PR-AUC｜Brier｜Top-k｜风险分级\n成本—损失价值｜五折空间留出", TINT, GRAY),
    ]
    n = len(steps)
    top, bot, h = 0.965, 0.055, 0.125
    gap = (top - bot - n * h) / (n - 1)
    w = 0.88
    x = (1 - w) / 2
    for i, (title, sub, fc, ec) in enumerate(steps):
        y = top - h - i * (h + gap)
        box(ax2, x, y, w, h, title, sub, fc=fc, ec=ec, fs=10.5, subfs=8.8, lw=1.15)
        if i < n - 1:
            arrow(ax2, (0.5, y - 0.004), (0.5, y - gap + 0.004), color=SUB, lw=1.25, ms=11)
    save(fig, "fig_article_fig1_zh")


# ==================================================================
# 图2 时间对齐与训练—校准—测试隔离示意
# ==================================================================
def panel(ax, x, y, w, h, fc, ec, lw=1.25):
    """只画框不写文字，文字一律另行精确定位，避免叠压。"""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.016",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))


def fig2():
    fig = plt.figure(figsize=(12.4, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.40)
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("（a）特征旬与下一旬标签的时间对齐", fontsize=11.5, pad=8,
                 fontweight="bold", loc="left")

    # 一个防火年度的 23 个旬条
    x0, x1, ybar, hbar = 0.05, 0.95, 0.66, 0.115
    total = x1 - x0
    for k in range(23):
        xx = x0 + total * k / 23
        last = (k == 22)
        col = "#FBE9CF" if last else "#EDEDED"
        ec = ORANGE if last else "#9A9A9A"
        ax.add_patch(Rectangle((xx, ybar), total / 23 - 0.003, hbar,
                               facecolor=col, edgecolor=ec, linewidth=0.7, zorder=2))
    ax.text(x0, ybar + hbar + 0.055, "防火年度（11 月 1 日—次年 6 月 15 日），共 23 个旬",
            ha="left", va="bottom", fontsize=9.5, color=SUB)
    ax.text(x0 + total * 10.5 / 23, ybar + hbar / 2, "… 第 1—22 旬：特征旬 …",
            ha="center", va="center", fontsize=9.3, color=BLUE, zorder=3,
            fontweight="bold", bbox=dict(facecolor="#EDEDED", edgecolor="none", pad=1.4))
    ax.text(x0 + total * 22.5 / 23, ybar + hbar / 2, "23",
            ha="center", va="center", fontsize=9, color=INK, zorder=3, fontweight="bold")
    ax.annotate("第 23 旬仅用于形成标签，不作特征旬",
                xy=(x0 + total * 22.5 / 23, ybar + hbar),
                xytext=(0.72, 0.925), fontsize=9.2, color=SUB, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.1, mutation_scale=10))

    # 放大：特征旬 t → 下一旬 t+1（框内四行文字纵向分开）
    ty, th = 0.13, 0.34
    panel(ax, 0.05, ty, 0.40, th, TINT, BLUE)
    panel(ax, 0.56, ty, 0.39, th, TINT, ORANGE)
    ax.text(0.25, ty + th * 0.80, "特征旬 t（截至旬末可得）", ha="center", va="center",
            fontsize=10.4, fontweight="bold", color=BLUE, zorder=4)
    ax.text(0.25, ty + th * 0.52, "过去 7 日气象、植被与静态代理", ha="center", va="center",
            fontsize=8.9, color=SUB, zorder=4)
    ax.text(0.25, ty + th * 0.30, "NDVI 仅用已达可用时刻的最近一期", ha="center", va="center",
            fontsize=8.9, color=SUB, zorder=4)
    ax.text(0.755, ty + th * 0.80, "下一旬 t + 1（标签）", ha="center", va="center",
            fontsize=10.4, fontweight="bold", color=INK, zorder=4)
    ax.text(0.755, ty + th * 0.52, "旬内是否出现 ≥1 个林区网格火点", ha="center", va="center",
            fontsize=8.9, color=SUB, zorder=4)
    ax.text(0.755, ty + th * 0.30, "记为 1，否则记为 0", ha="center", va="center",
            fontsize=8.9, color=SUB, zorder=4)
    arrow(ax, (0.455, ty + th / 2), (0.552, ty + th / 2), color=INK, lw=1.5, ms=14)
    ax.text(0.503, ty + th / 2 + 0.045, "预测", ha="center", va="bottom",
            fontsize=9, color=SUB)
    ax.text(0.05, 0.025, "跨防火年度的旬序连接被排除；不使用当前旬及未来旬的信息。",
            ha="left", va="bottom", fontsize=9, color=MUTE)

    # (b) 时间隔离：三段色带，每行文字单独定位
    ax2 = fig.add_subplot(gs[1, 0]); ax2.axis("off")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.set_title("（b）训练—校准—测试的时间隔离", fontsize=11.5, pad=8,
                  fontweight="bold", loc="left")
    bands = [
        ("训练集", "2010—2018 防火年度", "负样本下采样后的建模样本",
         "79,690 单元｜正样本占比 18.0562%", TINT, BLUE, 0.47),
        ("校准集", "2019—2020 防火年度", "概率校准、业务先验估计与规则冻结",
         "115,896 完整单元｜正样本占比 2.0363%", TINT, ORANGE, 0.25),
        ("测试集", "2021—2023 防火年度", "仅用于最终评价，不参与任何规则选择",
         "173,844 完整单元｜正样本占比 2.0725%", TINT, GRAY, 0.28),
    ]
    xx = 0.045
    for name, yr, role, stat, fc, ec, w in bands:
        bw = w - 0.012
        cx = xx + bw / 2
        panel(ax2, xx, 0.29, bw, 0.40, fc, ec, lw=1.3)
        ax2.text(cx, 0.615, name, ha="center", va="center", fontsize=11.5,
                 fontweight="bold", color=INK, zorder=4)
        ax2.text(cx, 0.525, yr, ha="center", va="center", fontsize=9.4,
                 color=INK, zorder=4, fontweight="bold")
        ax2.text(cx, 0.435, role, ha="center", va="center", fontsize=8.6,
                 color=SUB, zorder=4)
        ax2.text(cx, 0.345, stat, ha="center", va="center", fontsize=8.4,
                 color=SUB, zorder=4)
        xx += w
    arrow(ax2, (0.045, 0.19), (0.955, 0.19), color=INK, lw=1.4, ms=13)
    ax2.text(0.5, 0.135, "时间方向（不可逆；按防火年度依次划分，避免未来信息泄漏）",
             ha="center", va="top", fontsize=9.4, color=SUB)
    ax2.text(0.045, 0.78, "三个子集在“网格 × 防火年度 × 特征旬”层面互不重叠；校准与测试保留下采样前的完整单元。",
             ha="left", va="center", fontsize=9.2, color=MUTE)
    save(fig, "fig_article_fig2_zh")


# ==================================================================
# 图3 五折空间留出 AUC 与 Top-20% 捕获率（AUC 面板纵轴自 0 起）
# ==================================================================
def fig3():
    sm = pd.read_csv(BASE + "/results/spatial_cv_metrics.csv")
    ss = pd.read_csv(BASE + "/results/spatial_cv_summary.csv")
    sx = sm[sm.model == "XGBoost"].sort_values("spatial_fold")
    ssx = ss[ss.model == "XGBoost"].iloc[0]
    folds = sx.spatial_fold.values
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    a = axes[0]
    a.plot(folds, sx.AUC, color=ORANGE, marker="o", linestyle="-", linewidth=1.9,
           markersize=6, zorder=3, label="各折 AUC")
    a.axhline(ssx.AUC_pooled, color=BLUE, linestyle="--", linewidth=1.4,
              label="汇总 AUC = %.4f" % ssx.AUC_pooled)
    a.axhline(0.5, color=MUTE, linestyle=":", linewidth=1.0,
              label="随机判别基线（AUC = 0.5）")
    a.set_ylabel("AUC"); a.set_ylim(0, 1.0)
    style_axes(a)
    a.legend(frameon=False, fontsize=9, loc="upper right")
    a.text(0.02, 0.96, "（a）", transform=a.transAxes, va="top", fontweight="bold", fontsize=11)
    b = axes[1]
    b.plot(folds, sx.Top_20_capture * 100, color=ORANGE, marker="o", linestyle="-",
           linewidth=1.9, markersize=6, zorder=3, label="各折 Top-20% 捕获率")
    b.axhline(ssx.Top_20_capture_pooled * 100, color=BLUE, linestyle="--", linewidth=1.4,
              label="汇总捕获率 = %.2f%%" % (ssx.Top_20_capture_pooled * 100))
    b.axhline(20, color=MUTE, linestyle=":", linewidth=1.0,
              label="随机排序基线（20%）")
    b.set_ylabel("Top-20% 捕获率 / %"); b.set_ylim(0, 80)
    style_axes(b)
    b.legend(frameon=False, fontsize=9, loc="upper right")
    b.text(0.02, 0.96, "（b）", transform=b.transAxes, va="top", fontweight="bold", fontsize=11)
    for ax in axes:
        ax.set_xlabel("空间留出折")
        ax.set_xticks([1, 2, 3, 4, 5])
    fig.tight_layout()
    save(fig, "fig_article_fig3_zh")


# ==================================================================
# 图4 三类模型 ROC 与 PR 曲线（三色 + 线型区分）
# ==================================================================
def fig4():
    z = np.load(BASE + "/results/figure_cache.npz")
    y = z["yte"]
    models = [("XGBoost", "p_xgb", ORANGE, "-"), ("随机森林", "p_rf", BLUE, "--"),
              ("逻辑回归", "p_lr", GRAY, "-.")]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
    a = axes[0]
    a.plot([0, 1], [0, 1], color=MUTE, linestyle="--", linewidth=1.1, label="随机判别")
    for name, key, col, ls in models:
        p = z[key].astype(np.float64)
        fpr, tpr, _ = roc_curve(y, p)
        a.plot(fpr, tpr, color=col, linestyle=ls, linewidth=1.8,
               label="%s（AUC = %.4f）" % (name, roc_auc_score(y, p)))
    a.set_xlabel("假阳性率"); a.set_ylabel("真阳性率"); a.set_xlim(0, 1); a.set_ylim(0, 1)
    style_axes(a); a.legend(frameon=False, fontsize=8.8, loc="lower right")
    a.text(0.02, 0.96, "（a）ROC 曲线", transform=a.transAxes, va="top", fontweight="bold", fontsize=11)
    b = axes[1]
    prior = y.mean()
    b.axhline(prior, color=MUTE, linestyle="--", linewidth=1.1,
              label="评价期实际发生率 = %.4f" % prior)
    for name, key, col, ls in models:
        p = z[key].astype(np.float64)
        pr, rc, _ = precision_recall_curve(y, p)
        b.plot(rc, pr, color=col, linestyle=ls, linewidth=1.8,
               label="%s（PR-AUC = %.4f）" % (name, average_precision_score(y, p)))
    b.set_xlabel("召回率"); b.set_ylabel("精确率"); b.set_xlim(0, 1); b.set_ylim(0, 0.32)
    style_axes(b); b.legend(frameon=False, fontsize=8.8, loc="upper right")
    b.text(0.02, 0.96, "（b）PR 曲线", transform=b.transAxes, va="top", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "fig_article_fig4_zh")


# ==================================================================
# 图5 校准前后可靠性曲线
# ==================================================================
def fig5():
    d = pd.read_csv(BASE + "/results/calibration_bins.csv")
    pre = d[d["状态"] == "校准前"]; post = d[d["状态"] == "校准后"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
    for ax, sub, title, xmax in [(axes[0], pre, "（a）校准前", 0.80),
                                  (axes[1], post, "（b）校准后", 0.080)]:
        xp = sub["平均预测概率"].values; yp = sub["实际起火频率"].values
        ax.plot([0, xmax], [0, xmax], color=MUTE, linestyle="--", linewidth=1.2, label="完全校准（y = x）")
        ax.plot(xp, yp, color=ORANGE, marker="o", markersize=5.5, linewidth=1.7, label="8 个等频分箱")
        ax.set_xlim(0, xmax); ax.set_ylim(0, xmax * 1.02)
        ax.set_xlabel("平均预测概率"); ax.set_ylabel("实际阳性单元比例")
        style_axes(ax)
        ax.set_title(title, fontsize=11.5, pad=9, fontweight="bold")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_article_fig5_zh")


# ==================================================================
# 图6 Top-k 捕获曲线（三色 + 线型/标记区分，无红/绿/紫）
# ==================================================================
def fig6():
    d = pd.read_csv(BASE + "/results/baseline_capture.csv")
    items = [
        ("XGBoost全变量", "XGBoost（全变量，校准后）", BLUE,   "-",  "o", 2.3),
        ("历史网格火点频率", "历史网格火点频率",           ORANGE, "--", "s", 1.7),
        ("网格×旬历史气候频率", "网格×旬历史气候频率",     GRAY,   "-.", "^", 1.7),
        ("上一旬火点状态", "上一旬火点状态",               BLUE,   ":",  "D", 1.7),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for name, lab, col, ls, mk, lw in items:
        s = d[d.model == name].sort_values("top_frac")
        ax.plot(s.top_frac * 100, s.capture * 100, marker=mk, markersize=5.5,
                linestyle=ls, linewidth=lw, color=col, label=lab, zorder=3)
    ax.plot([0, 30], [0, 30], color=MUTE, linestyle="--", linewidth=1.2, label="随机排序")
    ax.set_xlabel("标记为最高风险单元比例 / %")
    ax.set_ylabel("火点阳性单元捕获率 / %")
    ax.set_xlim(0, 30); ax.set_ylim(0, 80); ax.set_xticks([0, 5, 10, 20, 30])
    style_axes(ax); ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_article_fig6_zh")


# ==================================================================
# 图7 成本—损失相对经济价值曲线
#   (a) 纵轴自 0 起；(b) 纵轴含负值并明确标注，V=0 虚线自解释
# ==================================================================
def fig7():
    curve = pd.read_csv(CL + "/rev_curve.csv")
    z = np.load(BASE + "/results/spatial_cv_predictions.npz")
    ys, folds, ps = z["yte"], z["spatial_fold"], z["p_xgb_cal"].astype(np.float64)
    alphas = np.round(np.linspace(0.001, 0.30, 1200), 6)
    n_all, pos_all = len(ys), int(ys.sum())

    def rev_at(yy, pp, alpha):
        nn = len(yy); pii = yy.sum() / nn
        flag = pp > alpha
        hit = int((yy[flag] == 1).sum()); fa = int(flag.sum() - hit); miss = int(yy.sum() - hit)
        e_f = (alpha * (hit + fa) + miss) / nn
        e_c = min(alpha, pii); e_p = alpha * pii
        den = e_c - e_p
        return np.nan if den <= 1e-15 else (e_c - e_f) / den

    sp = np.array([rev_at(ys, ps, a) for a in alphas])
    pi_test = 3603 / 173844
    a_star = 0.020701

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    styles = {
        "XGBoost全变量":       (BLUE,   "-",  2.3),
        "历史网格火点频率":     (ORANGE, "--", 1.7),
        "网格×旬历史气候频率": (GRAY,   "-.", 1.7),
        "上一旬火点状态":       (BLUE,   ":",  1.7),
    }
    labs = {"XGBoost全变量": "XGBoost（全变量）", "历史网格火点频率": "历史网格火点频率",
            "网格×旬历史气候频率": "网格×旬历史气候频率", "上一旬火点状态": "上一旬火点状态"}
    a = axes[0]
    for name in ["XGBoost全变量", "历史网格火点频率", "网格×旬历史气候频率", "上一旬火点状态"]:
        g = curve[curve["method"] == name].dropna(subset=["REV"])
        col, ls, lw = styles[name]
        a.plot(g.alpha, g.REV, lw=lw, color=col, linestyle=ls, label=labs[name])
    a.axhline(0, color=INK, lw=1.0, ls="--", alpha=0.6)
    a.axhline(1, color=GRAY, lw=1.2, ls=":", label="完美预报上界（V = 1）")
    a.axvline(pi_test, color=BLUE, lw=1.1, ls="-.", alpha=0.75,
              label="评价期实际发生率 π_test = 0.0207")
    a.set_xlim(0, 0.16); a.set_ylim(0, 1.05)
    a.set_xlabel("成本—损失比 α = C/L"); a.set_ylabel("相对经济价值 V")
    style_axes(a); a.legend(fontsize=8, loc="upper right", framealpha=0.92)
    a.text(0.02, 0.96, "（a）2021—2023 时间外推测试期", transform=a.transAxes,
           va="top", fontweight="bold", fontsize=10.5)

    b = axes[1]
    g = curve[curve["method"] == "XGBoost全变量"].dropna(subset=["REV"])
    b.plot(g.alpha, g.REV, lw=2.1, color=BLUE, linestyle="-", label="时间外推（XGBoost）")
    ok = np.isfinite(sp)
    b.plot(alphas[ok], sp[ok], lw=1.9, color=ORANGE, linestyle="--", label="空间留出汇总（XGBoost）")
    b.axhline(0, color=INK, lw=1.0, ls="--", alpha=0.6, label="V = 0（虚线）")
    b.axvline(pi_test, color=BLUE, lw=1.1, ls="-.", alpha=0.75,
              label="评价期实际发生率 π_test = 0.0207")
    b.set_xlim(0, 0.16); b.set_ylim(-0.08, 0.55)
    b.set_xlabel("成本—损失比 α = C/L"); b.set_ylabel("相对经济价值 V")
    style_axes(b); b.legend(fontsize=8.4, loc="upper right", framealpha=0.95)
    b.text(0.02, 0.96, "（b）时间外推与空间留出对比", transform=b.transAxes,
           va="top", fontweight="bold", fontsize=10.5)
    b.text(0.98, 0.03, "纵轴含负值；虚线为 V = 0", transform=b.transAxes,
           ha="right", va="bottom", fontsize=8.6, color=SUB, zorder=6,
           bbox=dict(facecolor=BG, edgecolor="none", alpha=0.85, pad=1.5))
    b.plot([a_star], [0.1072], marker="o", markersize=6, color=ORANGE,
           markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    fig.tight_layout()
    save(fig, "fig_article_fig7_zh")


if __name__ == "__main__":
    print("中文审阅版图件输出目录:", OUT)
    for fn in [fig1, fig2, fig3, fig4, fig5, fig6, fig7]:
        print("绘制", fn.__name__)
        fn()
    print("\n共 7 张图，仅输出 7 个 PNG 文件（不输出 PDF）。")
