"""Static figures for the README (docs/figures/*.png), drawn from results/*.json."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import RESULTS, SITE
from .taxonomy import MIT_DOMAINS

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
OUT = SITE / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def domains(tr):
    bd = tr["by_domain"]
    order = [d for d in MIT_DOMAINS.values() if d in bd["series"]]
    yrs = np.array(bd["years"])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bottom = np.zeros(len(yrs))
    for i, d in enumerate(order):
        v = np.array(bd["series"][d])
        ax.bar(yrs, v, bottom=bottom, color=PAL[i], label=d, width=0.75, edgecolor="#fcfcfb", linewidth=1)
        bottom += v
    ax.set_title("Canonical AI incidents per year by MIT risk domain (current year partial)", loc="left", color=INK)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    fig.savefig(OUT / "incidents_by_domain.png")
    plt.close(fig)


def heat(tr):
    fm = tr["by_failure_mode"]
    yrs = [y for y in fm["years"] if y >= 2015]
    idx = [fm["years"].index(y) for y in yrs]
    keys = sorted(fm["series"], key=lambda k: -sum(fm["series"][k]))
    M = np.array([[fm["series"][k][i] / max(fm["totals"][i], 1) for i in idx] for k in keys])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(M, cmap="Blues", aspect="auto", vmin=0, vmax=max(0.35, M.max()))
    ax.set_xticks(range(len(yrs)), yrs, rotation=45)
    ax.set_yticks(range(len(keys)), [k.replace("_", " ") for k in keys])
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            if M[r, c] >= 0.01:
                ax.text(c, r, f"{100 * M[r, c]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if M[r, c] > 0.2 else INK)
    ax.set_title("Failure-mode share of each year's incidents (%; multi-label)", loc="left", color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(OUT / "failure_mode_heatmap.png")
    plt.close(fig)


def llm_share(tr):
    ls = tr["llm_share_by_year"]
    y = np.array(ls["years"])
    sel = y >= 2016
    s = np.array(ls["share"])[sel]
    ci = np.array(ls["ci95"])[sel]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.fill_between(y[sel], ci[:, 0], ci[:, 1], color=PAL[0], alpha=0.15, linewidth=0)
    ax.plot(y[sel], s, color=PAL[0], lw=2, marker="o", ms=4)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.grid(True, color=GRID)
    ax.set_title("Share of incidents involving a frontier-LLM vendor (95% Wilson CI)", loc="left", color=INK, fontsize=10)
    fig.savefig(OUT / "llm_share.png")
    plt.close(fig)


def validation(val, model):
    v = val.get(model)
    if not v:
        return
    hh = val.get("human_human_baseline_cset", {})
    rows = [("MIT domain (7)", v["mit"]["domain_7class"]["kappa"], None),
            ("MIT subdomain (24)", v["mit"]["subdomain"]["kappa"], None),
            ("MIT entity", v["mit"]["entity"]["kappa"], None),
            ("MIT intent", v["mit"]["intent"]["kappa"], None),
            ("MIT timing", v["mit"]["timing"]["kappa"], None),
            ("CSET sector", v["cset"]["sector_coarse"]["kappa"], hh.get("sector_coarse", {}).get("weighted_mean_kappa")),
            ("CSET intentional harm", v["cset"]["intentional_harm"]["kappa"], hh.get("intentional_harm", {}).get("weighted_mean_kappa")),
            ("AIAAIC deepfake tag", v["aiaaic"]["deepfake"]["kappa"], None),
            ("AIAAIC privacy tag", v["aiaaic"]["privacy"]["kappa"], None),
            ("AIAAIC fairness tag", v["aiaaic"]["bias"]["kappa"], None)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ys = np.arange(len(rows))[::-1]
    for y, (name, k, h) in zip(ys, rows):
        ax.errorbar(k["value"], y, xerr=[[k["value"] - k["ci95"][0]], [k["ci95"][1] - k["value"]]],
                    fmt="o", color=PAL[0], ms=6, capsize=3)
        if h is not None:
            ax.plot(h, y, marker="D", color=PAL[1], ms=6, linestyle="none")
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlim(-0.05, 1)
    ax.axvline(0.6, color=GRID, lw=1, ls="--")
    ax.xaxis.grid(True, color=GRID)
    ax.set_xlabel("Cohen's κ (95% bootstrap CI)")
    ax.plot([], [], "o", color=PAL[0], label=f"Gemini ({model}) vs reference")
    ax.plot([], [], "D", color=PAL[1], label="CSET annotator vs annotator")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Classifier agreement with existing labels", loc="left", color=INK)
    fig.savefig(OUT / "validation_kappa.png")
    plt.close(fig)


def signals(sig):
    f = [s for s in sig["signals"] if s["dimension"] in ("failure_mode", "model_family", "sector")]
    f = sorted(f, key=lambda s: -s["z"])[:14][::-1]
    if not f:
        return
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ys = np.arange(len(f))
    ax.barh(ys, [s["share_ratio"] for s in f], color=[PAL[1] if s["flag"] else "#c3c2b7" for s in f], height=0.6)
    ax.axvline(1, color=INK2, lw=1)
    ax.set_yticks(ys, [f"{s['dimension'].replace('_', ' ')}: {s['label'].replace('_', ' ')}" for s in f], fontsize=8)
    for y, s in zip(ys, f):
        ax.text(s["share_ratio"], y, f"  {s['recent']} vs {s['baseline']}  z={s['z']}", va="center", fontsize=7, color=INK2)
    ax.set_xlabel("Share ratio: last 90 days vs prior 365 days")
    ax.set_title(f"Emerging signals as of {sig['as_of']} (orange = flagged)", loc="left", color=INK)
    fig.savefig(OUT / "emerging_signals.png")
    plt.close(fig)


def main():
    tr = json.loads((RESULTS / "trends.json").read_text())
    sig = json.loads((RESULTS / "signals.json").read_text())
    val = json.loads((RESULTS / "validation.json").read_text()) if (RESULTS / "validation.json").exists() else {}
    model = json.loads((RESULTS / "summary.json").read_text())["classifier"]
    domains(tr), heat(tr), llm_share(tr), validation(val, model), signals(sig)
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
