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
from concurrent.futures import ThreadPoolExecutor
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
    y_true, y_pred = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
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
            mm = m[(m[f"{col}_a"].astype(str).isin(["", "nan"]) == False) &
                   (m[f"{col}_b"].astype(str).isin(["", "nan"]) == False)]
            if len(mm) >= 10:
                res[key].append({"n": int(len(mm)),
                                 "kappa": round(float(cohen_kappa_score(
                                     mm[f"{col}_a"].astype(str), mm[f"{col}_b"].astype(str))), 3)})
    summary = {}
    for k, v in res.items():
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
JUDGE_PROMPT = """Do these two records describe the SAME real-world AI incident (same event, same system, same harmed party)? A news story about the incident counts as the same incident. Different incidents involving the same company or same type of harm are NOT the same.
Answer with JSON {"same": true|false}.

Record A: {a}

Record B: {b}"""


def dedupe_audit(model: str = "gemini-3.5-flash", per_bin: int = 25) -> dict:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    rec = pd.read_parquet(PROC / "records.parquet").set_index("uid")
    pairs = pd.read_parquet(PROC / "dedupe_pairs.parquet")
    cache_p = LABELS / f"dedupe_judge_{model}.jsonl"
    cache = {(r["a"], r["b"]): r["same"] for r in read_jsonl(cache_p)}
    bins = [0.70, 0.76, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.93, 1.01]
    sample = []
    for kind in ("aiaaic-aiid", "news-incident"):
        p = pairs[(pairs.kind == kind) & pairs.year_ok]
        for lo, hi in zip(bins[:-1], bins[1:]):
            g = p[(p.sim >= lo) & (p.sim < hi)]
            sample.append(g.sample(min(per_bin, len(g)), random_state=0).assign(bin=f"{lo:.2f}-{hi:.2f}"))
    sample = pd.concat(sample)

    def judge(row):
        key = (row.a, row.b)
        if key in cache:
            return cache[key]
        ra, rb = rec.loc[row.a], rec.loc[row.b]
        prompt = JUDGE_PROMPT.replace("{a}", f"{ra.title} ({ra.date}). {ra.text[:600]}") \
                             .replace("{b}", f"{rb.title} ({rb.date}). {rb.text[:600]}")
        delay = 4
        for _ in range(8):
            try:
                r = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"))
                same = bool(json.loads(r.text).get("same"))
                with open(cache_p, "a") as f:
                    f.write(json.dumps({"a": row.a, "b": row.b, "same": same}) + "\n")
                return same
            except Exception as e:
                time.sleep(delay + random.random())
                delay = min(delay * 2, 90)
        return None

    with ThreadPoolExecutor(4) as ex:
        sample["same"] = list(ex.map(judge, sample.itertuples()))
    sample = sample[sample.same.notna()]
    out = {"judge_model": model, "by_bin": {}}
    for (kind, b), g in sample.groupby(["kind", "bin"]):
        out["by_bin"].setdefault(kind, {})[b] = {"n": int(len(g)), "frac_same": round(float(g.same.mean()), 3)}
    from .dedupe import T_INC, T_NEWS
    for kind, t in (("aiaaic-aiid", T_INC), ("news-incident", T_NEWS)):
        g = sample[sample.kind == kind]
        above = g[g.sim >= t]
        out[f"{kind}_precision_above_threshold_sampled"] = round(float(above.same.mean()), 3) if len(above) else None
        out[f"{kind}_missed_dupes_rate_below_threshold_sampled"] = round(float(g[g.sim < t].same.mean()), 3)
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
