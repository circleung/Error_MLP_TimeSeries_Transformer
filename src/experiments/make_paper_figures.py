"""Generate publication-quality figures from the committed accident analysis JSONs.

Reads every <out_root>/<cell>/{tail_analysis,variable_analysis,
ar_permutation_importance,turning_point_analysis}.json (out_root + cell list from
configs/error_mlp_accident.yaml, same source the aggregate_*.py scripts use) and
renders 5 figures (each as 300-dpi PNG + vector PDF) into <repo_root>/figures/.

    Run from src/ (has matplotlib via the transformer_env):
    NONINTERACTIVE=1 python experiments/make_paper_figures.py

Read-only on data. Deterministic (no randomness, no network). Idempotent: reruns
overwrite the same 10 files. NOTHING is trained or rolled out here.

Figures:
  fig1  Tail is the problem; selective gated correction fixes it (MAIN result).
  fig2  Why selective, not global (global beta monotonically worsens mean).
  fig3  Importance != error: input importance does not track output error (5/5).
  fig4  TF vs AR permutation importance per input channel (5/5).
  fig5  Turning-point failure-mode hypothesis refuted (H1 <1, H2 up at TP, H3 ~1).
"""

import os
import sys
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless / deterministic raster backend
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SRC_DIR)
FIG_DIR = os.path.join(REPO_ROOT, "figures")
sys.path.insert(0, SRC_DIR)
import utils  # noqa: E402

# ---------------------------------------------------------------------------
# Okabe-Ito colorblind-safe palette (fixed order; NOT the matplotlib cycle).
# ---------------------------------------------------------------------------
OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",   # bluish-green
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",  # reddish-purple
}
GRAY = "#999999"          # muted gray for the "baseline / before" series
BASELINE_C = GRAY
GATED_C = OI["blue"]      # "gated / after"
WORSE_C = OI["vermillion"]
REF_C = "#555555"         # reference-line / annotation ink
MINUS = "−"          # typographic minus for value labels

# 5-line Okabe-Ito order for Fig 2a (yellow dropped: illegible as a line on white).
LINE_COLORS = [OI["black"], OI["orange"], OI["skyblue"], OI["green"], OI["blue"]]
LINE_MARKERS = ["o", "s", "^", "D", "v"]


def set_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,   # embed TrueType -> selectable/editable text in PDF
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def ygrid(ax):
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)


def short(cell):
    return cell.replace("_", "-")


def signed_pct(x, nd=1):
    """Format a signed percentage with a typographic minus, e.g. '-33.5%'."""
    s = f"{abs(x):.{nd}f}%"
    return (MINUS if x < 0 else "+") + s


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    pdf = os.path.join(FIG_DIR, name + ".pdf")
    png = os.path.join(FIG_DIR, name + ".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all(out_root, cells):
    data = {}
    for c in cells:
        cdir = os.path.join(out_root, c)
        data[c] = {
            "tail": json.load(open(os.path.join(cdir, "tail_analysis.json"))),
            "var": json.load(open(os.path.join(cdir, "variable_analysis.json"))),
            "ar": json.load(open(os.path.join(cdir, "ar_permutation_importance.json"))),
            "tp": json.load(open(os.path.join(cdir, "turning_point_analysis.json"))),
        }
    return data


# ---------------------------------------------------------------------------
# Fig 1 - Tail is the problem; selective gated correction fixes it (MAIN).
#   (a) baseline p99 vs gated-operating-point p99 per cell (+ % p99 reduction)
#   (b) baseline mean vs gated-operating-point mean per cell (mean stays flat)
#   Cells ordered by descending p99 reduction (computed, not hardcoded).
# tail_analysis.json: baseline.{p99,mean}, operating_point_strict.{p99,mean,p99_reduction_pct}
# ---------------------------------------------------------------------------
def fig1(data, cells):
    rows = []
    for c in cells:
        t = data[c]["tail"]
        b, op = t["baseline"], t["operating_point_strict"]
        rows.append({
            "cell": c,
            "base_p99": b["p99"], "op_p99": op["p99"],
            "base_mean": b["mean"], "op_mean": op["mean"],
            "p99_red": op["p99_reduction_pct"],
            "mean_delta_pct": (op["mean"] - b["mean"]) / b["mean"] * 100.0,
        })
    rows.sort(key=lambda r: r["p99_red"], reverse=True)  # descending effect size
    labels = [short(r["cell"]) for r in rows]
    x = np.arange(len(rows))
    w = 0.38

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.0, 3.3))

    # (a) p99
    b0 = ax0.bar(x - w / 2, [r["base_p99"] for r in rows], w,
                 color=BASELINE_C, label="baseline")
    b1 = ax0.bar(x + w / 2, [r["op_p99"] for r in rows], w,
                 color=GATED_C, label="gated (operating point)")
    ymax = max(r["base_p99"] for r in rows)
    for xi, r in zip(x, rows):
        top = max(r["base_p99"], r["op_p99"])
        ax0.annotate(f"{MINUS}{r['p99_red']:.1f}%", (xi, top),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=7.5, color=OI["black"])
    ax0.set_ylabel("per-step error p99 (scaled)")
    ax0.set_title("(a) Tail (p99) is cut by gating", loc="left")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=20, ha="right")
    ax0.set_ylim(0, ymax * 1.18)
    ax0.legend(frameon=False, loc="upper right")
    ygrid(ax0)

    # (b) mean
    ax1.bar(x - w / 2, [r["base_mean"] for r in rows], w, color=BASELINE_C)
    ax1.bar(x + w / 2, [r["op_mean"] for r in rows], w, color=GATED_C)
    mmax = max(r["base_mean"] for r in rows)
    for xi, r in zip(x, rows):
        top = max(r["base_mean"], r["op_mean"])
        ax1.annotate(signed_pct(r["mean_delta_pct"]), (xi, top),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=7.5, color=REF_C)
    ax1.set_ylabel("per-step error mean (scaled)")
    ax1.set_title("(b) Mean is not regressed", loc="left")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylim(0, mmax * 1.20)
    ygrid(ax1)

    fig.suptitle("Selective gated correction cuts the p99 error tail while keeping "
                 "the mean flat", fontsize=10, y=1.02)
    fig.tight_layout()
    return save(fig, "fig1_tail_gating")


# ---------------------------------------------------------------------------
# Fig 2 - Why selective, not global.
#   (a) mean micro-MAE vs global beta in {0,.25,.5,.75,1}; one line per cell.
#   (b) SBO beta=0.5 contrast: {baseline, global b=0.5, gated b=0.5,q=0.1} x {mean,p99}.
# tail_analysis.json: baseline.mean (=beta 0), global_beta[...].mean,
#   gated_pred["beta=0.5,q=0.1"], baseline/global/gated {mean,p99}
# ---------------------------------------------------------------------------
def fig2(data, cells):
    betas = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.0, 3.3))

    # (a) global-beta sweep
    for i, c in enumerate(cells):
        t = data[c]["tail"]
        ys = [t["baseline"]["mean"]]
        for b in betas[1:]:
            ys.append(t["global_beta"][f"beta={b}"]["mean"])
        ax0.plot(betas, ys, marker=LINE_MARKERS[i], color=LINE_COLORS[i],
                 markersize=4, linewidth=1.4, label=short(c))
    ax0.axvline(0.0, color=REF_C, linestyle=":", linewidth=0.9, zorder=0)
    ax0.plot([], [], color=REF_C, linestyle=":", linewidth=0.9,
             label="β=0: baseline")  # legend proxy for the baseline marker
    ax0.set_xlabel("global correction strength  β")
    ax0.set_ylabel("mean micro-MAE (scaled)")
    ax0.set_title("(a) Global β monotonically worsens the mean", loc="left")
    ax0.set_xticks(betas)
    ax0.legend(frameon=False, loc="upper left", ncol=1)
    ygrid(ax0)

    # (b) SBO beta=0.5 contrast (mean & p99)
    t = data["SBO"]["tail"]
    base = t["baseline"]
    glob = t["global_beta"]["beta=0.5"]
    gate = t["gated_pred"]["beta=0.5,q=0.1"]
    metrics = ["mean", "p99"]
    series = [("baseline", base, BASELINE_C),
              ("global β=0.5", glob, WORSE_C),
              ("gated β=0.5, q=0.1", gate, GATED_C)]
    xg = np.arange(len(metrics))
    w = 0.26
    for j, (name, d, col) in enumerate(series):
        vals = [d[m] for m in metrics]
        ax1.bar(xg + (j - 1) * w, vals, w, color=col, label=name)
    # direction arrows vs baseline (up = worse, down = better)
    for k, m in enumerate(metrics):
        bv = base[m]
        for j, (name, d, col) in enumerate(series):
            if j == 0:
                continue
            dv = (d[m] - bv) / bv * 100.0
            arrow = "↑" if dv > 0 else "↓"
            ax1.annotate(f"{arrow}{signed_pct(dv, 0)}",
                         (xg[k] + (j - 1) * w, d[m]),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=7,
                         color=(WORSE_C if dv > 0 else OI["black"]))
    ax1.set_xticks(xg)
    ax1.set_xticklabels(["mean", "p99"])
    ax1.set_ylabel("SBO per-step error (scaled)")
    ax1.set_title("(b) At equal β=0.5: global worse, gated better", loc="left")
    ax1.set_ylim(0, glob["p99"] * 1.22)
    ax1.legend(frameon=False, loc="upper left")
    ygrid(ax1)

    fig.suptitle("Uniform global correction hurts; selective (gated) correction is "
                 "the only win", fontsize=10, y=1.02)
    fig.tight_layout()
    return save(fig, "fig2_global_vs_selective")


# ---------------------------------------------------------------------------
# Fig 3 - Importance != error (divergence), 5/5 cells.
#   scatter per cell: x = per-variable permutation importance (continuous, 10 pts),
#   y = per-variable output p99. Annotate Spearman rho(imp, p99) matching the axes.
# variable_analysis.json: link_importance_error.importance_continuous (== TF dMAE
#   for the 10 continuous vars), output_error.per_out_p99, continuous_cols.
# ---------------------------------------------------------------------------
def fig3(data, cells):
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6))
    axes = axes.ravel()
    for ax, c in zip(axes, cells):
        v = data[c]["var"]
        imp = np.array(v["link_importance_error"]["importance_continuous"], float)
        p99 = np.array(v["output_error"]["per_out_p99"], float)
        names = v["continuous_cols"]
        rho = spearmanr(imp, p99).correlation

        # points to label: most-important input, hardest output, 2nd hardest output
        order_p99 = np.argsort(p99)[::-1]
        lab = []
        for idx in [int(np.argmax(imp)), int(order_p99[0]), int(order_p99[1])]:
            if idx not in lab:
                lab.append(idx)
            if len(lab) == 3:
                break
        base_pts = [i for i in range(len(imp)) if i not in lab]
        ax.scatter(imp[base_pts], p99[base_pts], s=22, color=OI["blue"],
                   alpha=0.85, edgecolor="none", zorder=3)
        ax.scatter(imp[lab], p99[lab], s=30, color=OI["vermillion"],
                   edgecolor="black", linewidth=0.4, zorder=4)
        for k, idx in enumerate(lab):
            off, va = ((3, 3), "bottom") if k < 2 else ((3, -9), "top")
            ax.annotate(names[idx], (imp[idx], p99[idx]),
                        xytext=off, textcoords="offset points",
                        va=va, fontsize=6.5, color=OI["black"])
        ax.set_title(f"{short(c)}   ρ = {rho:+.2f}", loc="left", fontsize=9)
        ygrid(ax)
        ax.margins(0.12)

    for ax in axes[len(cells):]:
        ax.axis("off")
    fig.supxlabel("input permutation importance  ΔMAE (continuous vars)",
                  fontsize=9)
    fig.supylabel("output error  p99 (per output var)", fontsize=9)
    fig.suptitle("Input importance does not track output error "
                 "(no positive relation in 5/5 cells)", fontsize=10, y=1.00)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    return save(fig, "fig3_importance_error_divergence")


# ---------------------------------------------------------------------------
# Fig 4 - TF vs AR permutation importance per input channel.
#   scatter per cell: x = TF dMAE, y = AR dMAE (all channels); y=x dashed ref.
#   Annotate Spearman(all channels). Highlight ZWRB(6) (drops) and PSGGEN(1) (rises).
# ar_permutation_importance.json: tf_vs_ar.{feature_cols, tf_dMAE_overall,
#   ar_dMAE_overall, spearman_tf_ar_all_channels}
# ---------------------------------------------------------------------------
def fig4(data, cells):
    HIGHLIGHT = {"ZWRB(6)": OI["vermillion"], "PSGGEN(1)": OI["green"]}
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6))
    axes = axes.ravel()
    for ax, c in zip(axes, cells):
        tv = data[c]["ar"]["tf_vs_ar"]
        fc = tv["feature_cols"]
        tf = np.array(tv["tf_dMAE_overall"], float)
        ar = np.array(tv["ar_dMAE_overall"], float)
        rho = tv["spearman_tf_ar_all_channels"]
        hi_idx = {fc.index(n): col for n, col in HIGHLIGHT.items() if n in fc}

        base_pts = [i for i in range(len(fc)) if i not in hi_idx]
        ax.scatter(tf[base_pts], ar[base_pts], s=20, color=OI["blue"],
                   alpha=0.8, edgecolor="none", zorder=3)
        hi = max(tf.max(), ar.max()) * 1.05
        ax.plot([0, hi], [0, hi], linestyle="--", linewidth=0.9,
                color=REF_C, zorder=1)  # y = x reference
        for idx, col in hi_idx.items():
            ax.scatter([tf[idx]], [ar[idx]], s=34, color=col,
                       edgecolor="black", linewidth=0.4, zorder=4)
            ax.annotate(fc[idx], (tf[idx], ar[idx]),
                        xytext=(3, 3), textcoords="offset points",
                        fontsize=6.5, color=OI["black"])
        ax.set_title(f"{short(c)}   ρ = {rho:+.2f}", loc="left", fontsize=9)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ygrid(ax)

    for ax in axes[len(cells):]:
        ax.axis("off")
    # y=x proxy handle for a small legend in the spare panel
    axes[len(cells)].plot([], [], linestyle="--", color=REF_C, label="y = x")
    axes[len(cells)].scatter([], [], color=OI["vermillion"], edgecolor="black",
                             linewidth=0.4, label="ZWRB(6): drops under AR")
    axes[len(cells)].scatter([], [], color=OI["green"], edgecolor="black",
                             linewidth=0.4, label="PSGGEN(1): rises under AR")
    axes[len(cells)].legend(frameon=False, loc="center", fontsize=7.5)

    fig.supxlabel("teacher-forcing importance  ΔMAE", fontsize=9)
    fig.supylabel("autoregressive importance  ΔMAE", fontsize=9)
    fig.suptitle("Channel importance largely transfers TF→AR, with a few "
                 "reordered drivers", fontsize=10, y=1.00)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    return save(fig, "fig4_tf_vs_ar")


# ---------------------------------------------------------------------------
# Fig 5 - Turning-point hypothesis refuted.
#   (a) H1 tail-lift (<1 -> refuted, ref line 1.0)
#   (b) H2 directional hit-rate overall vs at-TP (TP higher -> refuted)
#   (c) H3 MISS/HIT downstream cum-err ratio @H=10 (~1.05-1.15, weak; ref line 1.0)
# turning_point_analysis.json: pooled.H1.lift_tail_at_tp, pooled.H2.{dir_hit_rate_overall,
#   dir_hit_rate_at_tp}, pooled.H3["10"].ratio_miss_over_hit
# ---------------------------------------------------------------------------
def fig5(data, cells):
    labels = [short(c) for c in cells]
    x = np.arange(len(cells))
    h1 = [data[c]["tp"]["pooled"]["H1"]["lift_tail_at_tp"] for c in cells]
    h2_all = [data[c]["tp"]["pooled"]["H2"]["dir_hit_rate_overall"] for c in cells]
    h2_tp = [data[c]["tp"]["pooled"]["H2"]["dir_hit_rate_at_tp"] for c in cells]
    h3 = [data[c]["tp"]["pooled"]["H3"]["10"]["ratio_miss_over_hit"] for c in cells]

    fig, (a, b, cc) = plt.subplots(1, 3, figsize=(7.0, 3.0))

    # (a) H1 tail lift
    a.bar(x, h1, 0.6, color=OI["blue"])
    a.axhline(1.0, color=REF_C, linestyle="--", linewidth=0.9)
    a.annotate("no lift (1.0)", (len(cells) - 1, 1.0), xytext=(0, 3),
               textcoords="offset points", ha="right", va="bottom",
               fontsize=7, color=REF_C)
    a.set_title("(a) H1: tail lift at TP", loc="left")
    a.set_ylabel("p99-tail share lift (TP / base rate)")
    a.set_xticks(x); a.set_xticklabels(labels, rotation=30, ha="right")
    a.set_ylim(0, 1.2)
    ygrid(a)

    # (b) H2 directional hit-rate overall vs at-TP
    w = 0.38
    b.bar(x - w / 2, h2_all, w, color=BASELINE_C, label="overall")
    b.bar(x + w / 2, h2_tp, w, color=GATED_C, label="at turning point")
    b.axhline(0.5, color=REF_C, linestyle=":", linewidth=0.8)
    b.set_title("(b) H2: directional hit-rate", loc="left")
    b.set_ylabel("directional hit-rate")
    b.set_xticks(x); b.set_xticklabels(labels, rotation=30, ha="right")
    b.set_ylim(0, 0.8)
    b.legend(frameon=False, loc="upper right")
    ygrid(b)

    # (c) H3 MISS/HIT downstream ratio @ H=10
    cc.bar(x, h3, 0.6, color=OI["blue"])
    cc.axhline(1.0, color=REF_C, linestyle="--", linewidth=0.9)
    cc.annotate("dashed = 1.0 (no blowup)", (0.5, 0.985),
                xycoords="axes fraction", ha="center", va="top",
                fontsize=7, color=REF_C)
    for xi, v in zip(x, h3):
        cc.annotate(f"{v:.2f}", (xi, v), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.5, color=OI["black"])
    cc.set_title("(c) H3: MISS/HIT cum-err @H=10", loc="left")
    cc.set_ylabel("downstream cum-err ratio (MISS / HIT)")
    cc.set_xticks(x); cc.set_xticklabels(labels, rotation=30, ha="right")
    cc.set_ylim(0, 1.4)
    ygrid(cc)

    fig.suptitle("Turning-point failure hypothesis refuted: H1<1, H2 higher at TP, "
                 "H3 only ~1.05–1.15", fontsize=10, y=1.03)
    fig.tight_layout()
    return save(fig, "fig5_turning_point")


def main():
    set_style()
    cfg = utils.load_config("error_mlp_accident")
    out_root = cfg["out_root"]
    cells = list(cfg["cells"].keys())
    data = load_all(out_root, cells)

    written = []
    for fn in (fig1, fig2, fig3, fig4, fig5):
        written.extend(fn(data, cells))

    print(f"figures dir: {FIG_DIR}")
    ok = True
    for p in written:
        sz = os.path.getsize(p) if os.path.exists(p) else 0
        ok = ok and sz > 0
        print(f"  {'OK ' if sz > 0 else 'MISSING'} {os.path.basename(p):40s} {sz:>8d} B")
    print(f"{len(written)} files written; all non-empty: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
