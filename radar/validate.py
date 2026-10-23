"""Stage 4: validate the Gemini classifier against existing labels, and audit dedupe thresholds.

Reference labels (all shipped in the AIID snapshot or the AIAAIC sheet):
  * MIT AI Risk Repository taxonomy on AIID incidents: domain, subdomain, entity, intent, timing
  * CSET AI Harm Taxonomy v1 (human annotators, peer reviewed): tangible harm, sector,
    intentional harm, lives lost. The per-annotator files give a human-human baseline.
  * AIAAIC editorial tags: technology (deepfake), ethical issue (privacy, fairness)
Reports Cohen's kappa / accuracy / macro-F1 with 95% bootstrap CIs (1,000 resamples).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

from .common import LABELS, PROC, RESULTS, read_jsonl, write_json

RNG = np.random.default_rng(0)
B = 1000


def boot(fn, *arrs):
    n = len(arrs[0])
    point = fn(*arrs)
    vals = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        try:
            vals.append(fn(*(a[idx] for a in arrs)))
        except ValueError:
            continue
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return {"value": round(float(point), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "n": int(n)}


def agreement(y_true, y_pred) -> dict:
    clean = lambda v: "NA" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
    y_true = np.array([clean(v) for v in y_true], dtype=object)
    y_pred = np.array([clean(v) for v in y_pred], dtype=object)
    return {
        "kappa": boot(lambda a, b: cohen_kappa_score(a, b), y_true, y_pred),
        "accuracy": boot(lambda a, b: accuracy_score(a, b), y_true, y_pred),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 3),
        "n": int(len(y_true)),
        "confusion": pd.crosstab(pd.Series(y_true, name="reference"),
                                 pd.Series(y_pred, name="gemini")).to_dict(),
    }


def snapshot_dir():
    meta = json.loads((PROC / "ingest_meta.json").read_text())
    from pathlib import Path
    return Path(meta["aiid_csv_dir"])


# ------------------------------------------------------------------ MIT
def mit_eval(lab: pd.DataFrame) -> dict:
    mit = pd.read_csv(snapshot_dir() / "classifications_MIT.csv", dtype=str).fillna("")
    mit["uid"] = "aiid:" + mit["Incident ID"]
    mit["ref_sub"] = mit["Risk Subdomain"].str.extract(r"^(\d\.\d)")[0]
    mit["ref_dom"] = mit["Risk Domain"].str.extract(r"^(\d)")[0]
    m = mit.merge(lab, on="uid", how="inner")
    m = m[m["ref_dom"].notna()]
    m["pred_dom"] = m["mit_subdomain"].str[0]
    out = {"domain_7class": agreement(m["ref_dom"], m["pred_dom"]),
           "subdomain": agreement(m["ref_sub"].fillna("?"), m["mit_subdomain"])}
    for col in ("Entity", "Intent", "Timing"):
        mm = m[m[col].isin(["AI", "Human", "Intentional", "Unintentional",
                            "Pre-deployment", "Post-deployment", "Other"])]
        out[col.lower()] = agreement(mm[col], mm[col.lower()])
    per_dom = {}
    for d, g in m.groupby("ref_dom"):
        per_dom[d] = {"n": int(len(g)), "recall": round(float((g["pred_dom"] == d).mean()), 3),
                      "precision": round(float((m.loc[m.pred_dom == d, "ref_dom"] == d).mean()), 3)
                      if (m.pred_dom == d).any() else None}
    out["per_domain"] = per_dom
    out["n_reference"] = int(len(mit))
    return out


# ------------------------------------------------------------------ CSET
CSET_SECTOR = [
    (r"information and communication", "ict"),
    (r"transportation", "transport"),
    (r"arts, entertainment", "entertainment"),
    (r"law enforcement", "law_enforcement"),
    (r"wholesale and retail", "retail"),
    (r"education", "education"),
    (r"human health", "health"),
    (r"financial", "finance"),
    (r"public administration|defence", "public_admin"),
]
OUR_SECTOR = {
    "technology_software": "ict", "social_media": "ict", "media_news": "ict",
    "consumer_chatbot": "ict", "transportation": "transport", "entertainment_arts": "entertainment",
    "policing_surveillance": "law_enforcement", "retail_ecommerce": "retail",
    "education": "education", "healthcare": "health", "finance": "finance",
    "government_public_services": "public_admin", "law_justice": "public_admin",
    "military_security": "public_admin", "elections_politics": "public_admin",
}
TANGIBLE = {"no tangible harm, near-miss, or issue": 0,
            "non-imminent risk of tangible harm (an issue) occurred": 1,
            "imminent risk of tangible harm (near miss) did occur": 2,
            "tangible harm definitively occurred": 3}


def cset_sector(s: str) -> str:
    first = s.split(",")[0].lower() if s else ""
    for pat, v in CSET_SECTOR:
        if re.search(pat, s.lower() if pat.startswith("arts") else first):
            return v
    return "other" if s and s != "unclear" else ""


def load_cset(name: str) -> pd.DataFrame:
    df = pd.read_csv(snapshot_dir() / name, dtype=str).fillna("")
    df["uid"] = "aiid:" + df["Incident ID"]
    df["sector_c"] = df["Sector of Deployment"].map(cset_sector)
    df["tangible"] = df["Tangible Harm"].map(TANGIBLE)
    df["intentional"] = df["Intentional Harm"].map(
        lambda s: "Intentional" if s.startswith("Yes") else ("Unintentional" if s.startswith("No") else ""))
    df["deaths"] = pd.to_numeric(df["Lives Lost"], errors="coerce")
    return df


def cset_eval(lab: pd.DataFrame) -> dict:
    ref = load_cset("classifications_CSETv1.csv")
    m = ref.merge(lab, on="uid")
    out = {}
    s = m[m.sector_c != ""]
    out["sector_coarse"] = agreement(s["sector_c"], s["sector"].map(OUR_SECTOR).fillna("other"))
    t = m[m.tangible.notna()]
    out["severity_vs_tangible_harm_spearman"] = boot(
        lambda a, b: spearmanr(a, b).statistic, t["tangible"].to_numpy(float),
        t["severity"].to_numpy(float))
    th = (t["tangible"] == 3).astype(int).to_numpy()
    out["severity_auc_tangible_harm_occurred"] = boot(
        lambda a, b: roc_auc_score(a, b), th, t["severity"].to_numpy(float))
    d = m[m.deaths.notna()]
    out["severity_auc_lives_lost"] = boot(
        lambda a, b: roc_auc_score(a, b), (d["deaths"] > 0).astype(int).to_numpy(),
        d["severity"].to_numpy(float))
    out["mean_severity_when_lives_lost"] = round(float(d.loc[d.deaths > 0, "severity"].mean()), 2)
    i = m[m.intentional != ""]
    out["intentional_harm"] = agreement(i["intentional"], i["intent"].replace("Other", "Unintentional"))
    return out


def human_baseline() -> dict:
    """Pairwise agreement between individual CSET annotators on the same incidents."""
    anns = [load_cset(f"classifications_CSETv1_Annotator-{k}.csv") for k in (1, 2, 3)]
    res = {"sector_coarse": [], "tangible_harm_4class": [], "intentional_harm": []}
    for a, b in combinations(anns, 2):
        m = a.merge(b, on="uid", suffixes=("_a", "_b"))
        for key, col in (("sector_coarse", "sector_c"), ("tangible_harm_4class", "tangible"),
                         ("intentional_harm", "intentional")):
            a_ = [str(x) for x in m[f"{col}_a"]]
            b_ = [str(x) for x in m[f"{col}_b"]]
            keep = [(x, y) for x, y in zip(a_, b_) if x not in ("", "nan", "None") and y not in ("", "nan", "None")]
            if len(keep) >= 10:
                res[key].append({"n": len(keep), "kappa": round(float(cohen_kappa_score(
                    [k[0] for k in keep], [k[1] for k in keep])), 3)})
    summary = {}
    for k, v in res.items():
        v = [x for x in v if x["kappa"] == x["kappa"]]  # drop undefined (single-class) pairs
        if v:
            w = np.array([x["n"] for x in v])
            summary[k] = {"pairs": v, "weighted_mean_kappa":
                          round(float(np.average([x["kappa"] for x in v], weights=w)), 3)}
    return summary


# ------------------------------------------------------------------ AIAAIC
def aiaaic_eval(lab: pd.DataFrame, canon: pd.DataFrame) -> dict:
    rec = pd.read_parquet(PROC / "records.parquet")
    a = rec[rec.source == "aiaaic"].copy()
    ex = a["extra"].map(json.loads)
    a["tech"] = ex.map(lambda e: e.get("technology", "").lower())
    a["eth"] = ex.map(lambda e: e.get("ethical_issue", "").lower())
    # Map each AIAAIC row to the canonical incident it ended up in.
    owner = {m: u for u, ms in zip(canon.uid, canon.members) for m in ms}
    a["canon"] = a["uid"].map(owner)
    m = a.merge(lab, left_on="canon", right_on="uid", suffixes=("", "_lab"))
    fm = m["failure_modes"].map(lambda x: set(x) if x is not None else set())
    out = {}
    for name, ref, pred in [
        ("deepfake", m["tech"].str.contains("deepfake"), fm.map(lambda s: "deepfake" in s)),
        ("privacy", m["eth"].str.contains("privacy"), fm.map(lambda s: "privacy" in s)),
        ("bias", m["eth"].str.contains("fairness|discriminat|bias"), fm.map(lambda s: "bias" in s)),
    ]:
        y, p = ref.to_numpy(int), pred.to_numpy(int)
        out[name] = {"kappa": boot(lambda a_, b_: cohen_kappa_score(a_, b_), y, p),
                     "f1": round(float(f1_score(y, p)), 3), "ref_positive_rate": round(float(y.mean()), 3),
                     "pred_positive_rate": round(float(p.mean()), 3), "n": int(len(y))}
    return out


# ------------------------------------------------------------------ dedupe audit
def dedupe_audit(per_bin: int = 30) -> dict:
    """How many true duplicates sit *below* the candidate-retrieval threshold (never judged)?

    Sample best-match pairs from similarity bins under the threshold, send them to the same
    batched LLM judge, and report the duplicate rate per bin. Above the threshold, report the
    judge's merge rate per bin from the production decisions.
    """
    from .dedupe import T_CAND_INC, T_CAND_NEWS, judge_pairs
    rec = pd.read_parquet(PROC / "records.parquet").reset_index(drop=True)
    pos = {u: i for i, u in enumerate(rec.uid)}
    pairs = pd.read_parquet(PROC / "dedupe_pairs.parquet")
    dec = pd.read_parquet(PROC / "dedupe_decisions.parquet")
    out = {"below_threshold": {}, "above_threshold": {}}
    sample = []
    for kind, t in (("aiaaic-aiid", T_CAND_INC), ("news-incident", T_CAND_NEWS)):
        p = pairs[(pairs.kind == kind) & pairs.year_ok]
        for lo, hi in ((t - 0.12, t - 0.06), (t - 0.06, t)):
            g = p[(p.sim >= lo) & (p.sim < hi)]
            g = g.sample(min(per_bin, len(g)), random_state=0).assign(bin=f"{lo:.2f}-{hi:.2f}", n_bin=len(g))
            sample.append(g)
        d = dec[dec.kind == kind]
        for lo, hi in ((t, t + 0.04), (t + 0.04, t + 0.08), (t + 0.08, 1.01)):
            g = d[(d.sim >= lo) & (d.sim < hi) & d.judged]
            out["above_threshold"].setdefault(kind, {})[f"{lo:.2f}-{min(hi, 1):.2f}"] = {
                "n": int(len(g)), "judged_same": round(float(g.same.mean()), 3) if len(g) else None}
    sample = pd.concat(sample)
    verdict = judge_pairs(rec, [(pos[a], pos[b]) for a, b in zip(sample.a, sample.b)])
    sample["same"] = [verdict.get((a, b)) for a, b in zip(sample.a, sample.b)]
    sample = sample[sample.same.notna()]
    est_missed = {}
    for (kind, b), g in sample.groupby(["kind", "bin"]):
        rate = float(g.same.astype(bool).mean())
        out["below_threshold"].setdefault(kind, {})[b] = {"n_judged": int(len(g)), "dup_rate": round(rate, 3),
                                                          "pairs_in_bin": int(g.n_bin.iloc[0])}
        est_missed[kind] = est_missed.get(kind, 0) + rate * int(g.n_bin.iloc[0])
    out["estimated_missed_duplicates_in_audited_bins"] = {k: int(round(v)) for k, v in est_missed.items()}
    sample.to_csv(RESULTS / "dedupe_audit_sample.csv", index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemini-3.1-flash-lite"])
    ap.add_argument("--dedupe-audit", action="store_true")
    args = ap.parse_args()
    canon = pd.read_parquet(PROC / "canonical.parquet")
    report = {"human_human_baseline_cset": human_baseline()}
    for model in args.models:
        p = PROC / f"labels_{model}.parquet"
        if not p.exists():
            print(f"skip {model}: no labels", file=sys.stderr)
            continue
        lab = pd.read_parquet(p)
        report[model] = {"mit": mit_eval(lab), "cset": cset_eval(lab), "aiaaic": aiaaic_eval(lab, canon)}
    if args.dedupe_audit:
        report["dedupe_audit"] = dedupe_audit()
    old = json.loads((RESULTS / "validation.json").read_text()) if (RESULTS / "validation.json").exists() else {}
    old.update(report)
    write_json(RESULTS / "validation.json", old, indent=2)
    for model in args.models:
        if model in report:
            r = report[model]
            print(model, "MIT domain kappa", r["mit"]["domain_7class"]["kappa"],
                  "| subdomain", r["mit"]["subdomain"]["kappa"]["value"],
                  "| CSET sector", r["cset"]["sector_coarse"]["kappa"]["value"])
    print(json.dumps(report.get("human_human_baseline_cset"), indent=1)[:800])
    if "dedupe_audit" in report:
        print(json.dumps(report["dedupe_audit"], indent=1))


if __name__ == "__main__":
    main()
