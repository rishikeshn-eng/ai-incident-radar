"""Stage 5: trend analysis + emerging-signals detector, and the JSON the dashboard reads.

Emerging signals compare the last 90 days with the preceding 365 days, using only records
with day-precision dates (AIID incident dates, HN/Reddit post dates). For each category the
test is on its *share* of all incidents in the window (so overall volume growth in AI news
does not flag everything): log share-ratio with a 0.5 continuity correction, z-score from the
delta-method SE, flagged when z >= 2 and >= 3 recent incidents.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .common import PROC, RESULTS, SITE, write_json
from .dedupe import cluster_topics
from .taxonomy import FAILURE_MODES, MIT_DOMAINS, MIT_SUBDOMAINS

from .classify import DEFAULT_MODEL as MODEL
WINDOW = 90
BASELINE = 365
LLM_FAMILIES = {"openai", "google", "anthropic", "meta", "xai", "microsoft", "deepseek", "mistral"}


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def load() -> pd.DataFrame:
    canon = pd.read_parquet(PROC / "canonical.parquet")
    lab = pd.read_parquet(PROC / f"labels_{MODEL}.parquet")
    canon["crow"] = np.arange(len(canon))
    df = canon.merge(lab, on="uid", how="inner")
    # Curated databases are incidents by construction; community posts must pass the filter.
    keep = df["source"].isin(["aiid", "aiaaic"]) | df["is_ai_incident"].astype(bool)
    emb = np.load(PROC / "canonical_emb.npy")
    df = df[keep].copy()
    # Re-cluster only real incidents, so community noise does not form its own topics.
    df["cluster"], df["cluster_name"] = cluster_topics(emb[df["crow"].to_numpy()], df["title"].tolist())
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["domain"] = df["mit_subdomain"].str[0].map(MIT_DOMAINS)
    df["llm_related"] = df["model_family"].isin(LLM_FAMILIES)
    df["failure_modes"] = df["failure_modes"].map(lambda x: list(x) if x is not None else [])
    df["harm_types"] = df["harm_types"].map(lambda x: list(x) if x is not None else [])
    return df


def by_year(df: pd.DataFrame, col: str, multi=False, years=None) -> dict:
    d = df[df.year.between(*years)]
    if multi:
        d = d.explode(col)
    t = pd.crosstab(d["year"].astype(int), d[col])
    totals = df[df.year.between(*years)].groupby(df.year.astype("Int64")).size()
    return {"years": [int(y) for y in t.index], "series": {c: t[c].astype(int).tolist() for c in t.columns},
            "totals": [int(totals.get(y, 0)) for y in t.index]}


def trends(df: pd.DataFrame, now: datetime) -> dict:
    last_full = now.year - 1
    yrs = (2012, now.year)
    out = {"years_range": list(yrs), "last_full_year": last_full}
    out["by_domain"] = by_year(df, "domain", years=yrs)
    out["by_failure_mode"] = by_year(df, "failure_modes", multi=True, years=yrs)
    out["by_sector"] = by_year(df, "sector", years=yrs)
    out["by_model_family"] = by_year(df, "model_family", years=yrs)
    out["by_harm_type"] = by_year(df, "harm_types", multi=True, years=yrs)
    out["by_source"] = by_year(df, "source", years=yrs)
    sev = df[df.year.between(*yrs)].groupby("year")["severity"].agg(["mean", "count", "std"])
    out["severity_by_year"] = {"years": [int(y) for y in sev.index],
                               "mean": sev["mean"].round(3).tolist(),
                               "ci95": [[round(m - 1.96 * s / math.sqrt(n), 3), round(m + 1.96 * s / math.sqrt(n), 3)]
                                        if n > 1 else [m, m] for m, s, n in zip(sev["mean"], sev["std"].fillna(0), sev["count"])]}
    # Share of incidents tied to LLM vendors, with Wilson CIs.
    g = df[df.year.between(*yrs)].groupby("year")["llm_related"].agg(["sum", "count"])
    out["llm_share_by_year"] = {"years": [int(y) for y in g.index],
                                "share": (g["sum"] / g["count"]).round(4).tolist(),
                                "ci95": [wilson(int(k), int(n)) for k, n in zip(g["sum"], g["count"])]}
    # Headline comparison: failure-mode share, last full year vs three years earlier.
    comp = {}
    for fm in FAILURE_MODES:
        row = {}
        for y in (last_full - 3, last_full):
            d = df[df.year == y]
            k = int(d["failure_modes"].map(lambda s: fm in s).sum())
            row[str(y)] = {"k": k, "n": int(len(d)), "share": round(k / max(len(d), 1), 4),
                           "ci95": wilson(k, len(d))}
        comp[fm] = row
    out["failure_mode_share_change"] = comp
    # Severity mix by domain (all years) for the stacked bar.
    sx = pd.crosstab(df["domain"], df["severity"])
    out["severity_by_domain"] = {d: sx.loc[d].astype(int).to_dict() for d in sx.index}
    return out


def share_test(r_k, r_n, b_k, b_n):
    p1 = (r_k + 0.5) / (r_n + 1)
    p0 = (b_k + 0.5) / (b_n + 1)
    lr = math.log(p1 / p0)
    se = math.sqrt(1 / (r_k + 0.5) - 1 / (r_n + 1) + 1 / (b_k + 0.5) - 1 / (b_n + 1))
    return lr, lr / se


def emerging(df: pd.DataFrame, now: datetime) -> dict:
    day = df[df["date_precision"].eq("day") & df["date_dt"].notna()].copy()
    w0 = pd.Timestamp(now.date()) - pd.Timedelta(days=WINDOW)
    b0 = w0 - pd.Timedelta(days=BASELINE)
    recent = day[(day.date_dt >= w0) & (day.date_dt <= pd.Timestamp(now.date()))]
    base = day[(day.date_dt >= b0) & (day.date_dt < w0)]
    signals = []

    def test(dim, values_fn, label_fn=lambda v: v):
        rv = recent.apply(values_fn, axis=1)
        bv = base.apply(values_fn, axis=1)
        cats = set(v for s in rv for v in s)
        for c in cats:
            rk = int(rv.map(lambda s: c in s).sum())
            bk = int(bv.map(lambda s: c in s).sum())
            lr, z = share_test(rk, len(recent), bk, len(base))
            ex = recent[rv.map(lambda s: c in s)].sort_values(["severity", "community_points"], ascending=False)
            signals.append({
                "dimension": dim, "category": c, "label": label_fn(c),
                "recent": rk, "baseline": bk, "recent_share": round(rk / max(len(recent), 1), 4),
                "baseline_share": round(bk / max(len(base), 1), 4),
                "share_ratio": round(math.exp(lr), 2), "z": round(z, 2),
                "flag": bool(z >= 2 and rk >= 3),
                "examples": [{"uid": r.uid, "title": r.title, "date": r.date, "url": r.url,
                              "severity": int(r.severity)} for r in ex.head(5).itertuples()],
            })

    test("failure_mode", lambda r: set(r["failure_modes"]))
    test("mit_subdomain", lambda r: {r["mit_subdomain"]},
         lambda c: f"{c} {MIT_SUBDOMAINS.get(c, '')}")
    test("model_family", lambda r: {r["model_family"]})
    test("sector", lambda r: {r["sector"]})
    test("cluster", lambda r: {int(r["cluster"])} if r["cluster"] >= 0 else set(),
         lambda c: df.loc[df.cluster == c, "cluster_name"].iloc[0])
    # Severe recent incidents are always surfaced regardless of statistics.
    severe = recent[recent.severity >= 4].sort_values(["severity", "community_points"], ascending=False)
    # Clusters that are new: >= 60% of their dated members fall inside the window.
    new_clusters = []
    for c, g in day[day.cluster >= 0].groupby("cluster"):
        if len(g) >= 4:
            frac = float((g.date_dt >= w0).mean())
            if frac >= 0.6:
                new_clusters.append({"cluster": int(c), "label": g.cluster_name.iloc[0],
                                     "n": int(len(g)), "recent_frac": round(frac, 2)})
    signals.sort(key=lambda s: (-s["flag"], -s["z"]))
    return {
        "as_of": now.date().isoformat(), "window_days": WINDOW, "baseline_days": BASELINE,
        "window": [w0.date().isoformat(), now.date().isoformat()],
        "baseline": [b0.date().isoformat(), w0.date().isoformat()],
        "n_recent": int(len(recent)), "n_baseline": int(len(base)),
        "recent_by_source": recent["source"].value_counts().to_dict(),
        "signals": signals, "flagged": [s for s in signals if s["flag"]],
        "severe_recent": [{"uid": r.uid, "title": r.title, "date": r.date, "url": r.url,
                           "severity": int(r.severity), "failure": r.primary_failure_mode,
                           "summary": r.summary} for r in severe.head(15).itertuples()],
        "new_clusters": new_clusters,
    }


def export_incidents(df: pd.DataFrame) -> list[dict]:
    rows = []
    for r in df.sort_values("date", ascending=False).itertuples():
        rows.append({
            "id": r.uid, "t": r.title, "s": r.summary, "d": r.date, "y": int(r.year) if r.year == r.year else None,
            "src": [str(x) for x in r.sources], "u": r.url, "dom": r.mit_subdomain, "fm": [str(x) for x in r.failure_modes],
            "pfm": r.primary_failure_mode, "sev": int(r.severity), "sec": r.sector,
            "mf": r.model_family, "sys": r.system_name, "ent": r.entity, "int": r.intent,
            "ht": [str(x) for x in r.harm_types], "cl": int(r.cluster), "cn": r.cluster_name,
            "dep": r.deployer[:80], "cm": int(r.community_mentions), "x": float(r.x), "yy": float(r.y),
        })
    return rows


def main():
    now = datetime.now(timezone.utc)
    df = load()
    tr = trends(df, now)
    em = emerging(df, now)
    meta = json.loads((PROC / "ingest_meta.json").read_text())
    dd = json.loads((PROC / "dedupe_stats.json").read_text())
    val = json.loads((RESULTS / "validation.json").read_text()) if (RESULTS / "validation.json").exists() else {}
    summary = {
        "generated_at": now.isoformat(timespec="seconds"), "aiid_snapshot": meta["aiid_snapshot"],
        "ingest_counts": meta["counts"], "reddit_status": meta.get("reddit_status"),
        "dedupe": dd, "n_incidents": int(len(df)), "classifier": MODEL,
        "by_source_canonical": df["source"].value_counts().to_dict(),
    }
    write_json(RESULTS / "trends.json", tr, indent=1)
    write_json(RESULTS / "signals.json", em, indent=1)
    write_json(RESULTS / "summary.json", summary, indent=1)
    write_json(SITE / "data" / "incidents.json", export_incidents(df))
    write_json(SITE / "data" / "trends.json", tr)
    write_json(SITE / "data" / "signals.json", em)
    write_json(SITE / "data" / "summary.json", summary)
    write_json(SITE / "data" / "validation.json", val)
    print(json.dumps(summary, indent=1)[:1500])
    print("flagged signals:")
    for s in em["flagged"][:20]:
        print(f"  {s['dimension']:14s} {s['label'][:50]:50s} {s['recent']:4d} vs {s['baseline']:4d}  x{s['share_ratio']}  z={s['z']}")


if __name__ == "__main__":
    main()
