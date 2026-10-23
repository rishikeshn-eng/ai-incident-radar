"""Stage 7: regenerate the README results section and STATUS.md from results/*.json.

Everything between <!-- RESULTS:START --> and <!-- RESULTS:END --> in README.md is rewritten,
so the numbers in the README always match the committed data.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from .common import PROC, RESULTS, ROOT

REPO = "https://github.com/rishikeshn-eng/ai-incident-radar"
PAGES = "https://rishikeshn-eng.github.io/ai-incident-radar/"


def k(o):
    return f"{o['value']:.2f} [{o['ci95'][0]:.2f}, {o['ci95'][1]:.2f}]"


def results_md() -> tuple[str, dict]:
    S = json.loads((RESULTS / "summary.json").read_text())
    V = json.loads((RESULTS / "validation.json").read_text())
    T = json.loads((RESULTS / "trends.json").read_text())
    G = json.loads((RESULTS / "signals.json").read_text())
    model = S["classifier"]
    v = V[model]
    hh = V.get("human_human_baseline_cset", {})
    dd = S["dedupe"]
    rec = pd.read_parquet(PROC / "records.parquet")
    lab = pd.read_parquet(PROC / f"labels_{model}.parquet")
    labelled = rec[rec.uid.isin(lab.uid)].source.value_counts().to_dict()
    total = rec.source.value_counts().to_dict()
    hn_meta = int(((rec.source == "hn") & rec.title.str.match(r"^(Show|Launch|Ask|Tell) HN", case=False)).sum())
    hn_eligible = total.get("hn", 0) - hn_meta
    complete = all(labelled.get(s, 0) >= (hn_eligible if s == "hn" else n) - 5 for s, n in total.items())

    L = []
    L.append("## Results\n")
    L.append(f"*Generated {S['generated_at'][:10]} from AIID snapshot `{S['aiid_snapshot']}`; classifier `{model}`.*\n")
    if not complete:
        L.append("> **Partial run.** The free-tier Gemini quota (500 requests/day, shared with other projects) "
                 "ran out before every record was classified. Unclassified records are left out of the "
                 "trends below and are picked up automatically by the next run. Coverage: "
                 + ", ".join(f"{s} {labelled.get(s, 0)}/{hn_eligible if s == 'hn' else n}" for s, n in total.items())
                 + ".\n")
    L.append("### Data\n")
    L.append("| Source | Records ingested | Classified | Canonical incidents (primary source) |")
    L.append("|---|---:|---:|---:|")
    for s in ("aiid", "aiaaic", "hn", "reddit"):
        if s in total or s == "reddit":
            L.append(f"| {dict(aiid='AI Incident Database', aiaaic='AIAAIC Repository', hn='Hacker News (Algolia)', reddit='Reddit')[s]} "
                     f"| {total.get(s, 0):,} | {labelled.get(s, 0):,} | {S['by_source_canonical'].get(s, 0):,} |")
    L.append(f"\n- Reddit: {S['reddit_status']}.")
    L.append(f"- {hn_meta} Show/Launch/Ask/Tell HN posts are skipped before classification (product launches and discussions, not incident reports). "
             f"Of the HN stories classified, only those the classifier marks `is_ai_incident` enter the analysis.")
    L.append(f"- **Dedupe:** {dd['candidate_pairs']} candidate pairs retrieved by embedding similarity; the LLM judge confirmed "
             f"{dd['candidate_pairs_judged_same']} ({dd['judge_coverage']:.0%} of candidates judged; the rest fell back to cosine ≥ {dd['thresholds']['T_FALLBACK']}). "
             f"**{dd['aiaaic_merged_into_aiid']} AIAAIC entries merged into AIID incidents**, so naive concatenation would double-count "
             f"about {dd['aiaaic_merged_into_aiid'] / max(total.get('aiaaic', 1), 1):.0%} of AIAAIC. "
             f"{dd['records']:,} records → {dd['canonical']:,} canonical incidents; {S['n_incidents']:,} enter the analysis.")
    if "dedupe_audit" in V:
        a = V["dedupe_audit"]
        parts = []
        for kind, bins in a.get("below_threshold", {}).items():
            parts.append(f"{kind}: " + ", ".join(f"cos {b} → {x['dup_rate']:.0%} duplicates (n={x['n_judged']})" for b, x in bins.items()))
        L.append(f"- **Dedupe audit** (pairs *below* the retrieval threshold, judged by the same LLM): " + "; ".join(parts)
                 + f". Estimated duplicates missed in those bins: {a.get('estimated_missed_duplicates_in_audited_bins')}.")

    L.append("\n### Classifier validation\n")
    L.append("![Validation](docs/figures/validation_kappa.png)\n")
    L.append("| Reference labels | Field | n | Gemini κ [95% CI] | Accuracy | Human–human κ |")
    L.append("|---|---|---:|---:|---:|---:|")
    m = v["mit"]
    for name, key in (("Risk domain (7)", "domain_7class"), ("Risk subdomain (24)", "subdomain"),
                      ("Entity (AI/Human/Other)", "entity"), ("Intent", "intent"), ("Timing", "timing")):
        L.append(f"| MIT (LLM-generated, see note) | {name} | {m[key]['n']} | {k(m[key]['kappa'])} | {m[key]['accuracy']['value']:.2f} | – |")
    c = v["cset"]
    L.append(f"| CSET v1 (human) | Sector, 10 coarse classes | {c['sector_coarse']['n']} | {k(c['sector_coarse']['kappa'])} | {c['sector_coarse']['accuracy']['value']:.2f} | {hh.get('sector_coarse', {}).get('weighted_mean_kappa', float('nan')):.2f} |")
    L.append(f"| CSET v1 (human) | Intentional harm | {c['intentional_harm']['n']} | {k(c['intentional_harm']['kappa'])} | {c['intentional_harm']['accuracy']['value']:.2f} | {hh.get('intentional_harm', {}).get('weighted_mean_kappa', float('nan')):.2f} |")
    L.append(f"| CSET v1 (human) | Severity vs tangible-harm level (Spearman ρ) | {c['severity_vs_tangible_harm_spearman']['n']} | {k(c['severity_vs_tangible_harm_spearman'])} | – | (4-class tangible harm κ = {hh.get('tangible_harm_4class', {}).get('weighted_mean_kappa', float('nan')):.2f}) |")
    L.append(f"| CSET v1 (human) | Severity AUROC: tangible harm occurred | {c['severity_auc_tangible_harm_occurred']['n']} | {k(c['severity_auc_tangible_harm_occurred'])} | – | – |")
    L.append(f"| CSET v1 (human) | Severity AUROC: lives lost > 0 | {c['severity_auc_lives_lost']['n']} | {k(c['severity_auc_lives_lost'])} | – | – |")
    a = v["aiaaic"]
    for name, key in (("Deepfake (technology tag)", "deepfake"), ("Privacy (ethical-issue tag)", "privacy"),
                      ("Fairness/bias (ethical-issue tag)", "bias")):
        L.append(f"| AIAAIC (editorial) | {name} | {a[key]['n']} | {k(a[key]['kappa'])} | F1 {a[key]['f1']:.2f} | – |")
    L.append("")
    L.append(f"- **Sector** agreement with CSET's human annotators (κ {c['sector_coarse']['kappa']['value']:.2f}) is in the range of CSET annotators agreeing with each other "
             f"(κ {hh.get('sector_coarse', {}).get('weighted_mean_kappa', float('nan')):.2f}). "
             f"**Severity** ranks incidents with deaths almost perfectly (AUROC {c['severity_auc_lives_lost']['value']:.2f}; mean severity {c['mean_severity_when_lives_lost']} when lives were lost), "
             f"but correlates only moderately with CSET's 4-level tangible-harm scale (ρ {c['severity_vs_tangible_harm_spearman']['value']:.2f}).")
    L.append(f"- **Intentional harm is the weak field** (κ {c['intentional_harm']['kappa']['value']:.2f} vs human–human {hh.get('intentional_harm', {}).get('weighted_mean_kappa', float('nan')):.2f}). "
             "Part of this is a construct mismatch: CSET asks whether the *system was designed* to harm, while the radar's `intent` asks whether *anyone* intended the harm, so a deepfake scam counts as intentional here but usually not in CSET.")
    L.append("- **About the MIT reference:** the MIT risk-domain labels shipped in the AIID snapshot were themselves produced by an LLM pipeline "
             "(MIT AI Incident Tracker, Claude Sonnet, about 80% consistency with expert ratings per its authors). Agreement with them measures consistency between two model-based "
             "classifiers, not accuracy. The CSET rows are the human ground truth.")
    L.append(f"- Full confusion matrices and per-domain precision/recall: [`results/validation.json`](results/validation.json).")

    L.append("\n### Trends\n")
    L.append("![Failure modes by year](docs/figures/failure_mode_heatmap.png)\n")
    ly = T["last_full_year"]
    ch = T["failure_mode_share_change"]
    rows = []
    for fm, d in ch.items():
        a0, a1 = d[str(ly - 3)], d[str(ly)]
        rows.append((fm, a0, a1, a1["share"] - a0["share"]))
    rows.sort(key=lambda r: -r[3])
    L.append(f"Share of incidents tagged with each failure mode, {ly - 3} vs {ly} (95% Wilson CIs). Shares, not counts, because coverage grew.\n")
    L.append(f"| Failure mode | {ly - 3} | {ly} | Δ share |")
    L.append("|---|---:|---:|---:|")
    for fm, a0, a1, dlt in rows:
        L.append(f"| {fm.replace('_', ' ')} | {a0['share']:.1%} [{a0['ci95'][0]:.1%}–{a0['ci95'][1]:.1%}] ({a0['k']}/{a0['n']}) "
                 f"| {a1['share']:.1%} [{a1['ci95'][0]:.1%}–{a1['ci95'][1]:.1%}] ({a1['k']}/{a1['n']}) | {dlt:+.1%} |")
    ls = T["llm_share_by_year"]
    yi = {y: i for i, y in enumerate(ls["years"])}
    if ly in yi and (ly - 3) in yi:
        L.append(f"\n![LLM share](docs/figures/llm_share.png)\n\nIncidents tied to a frontier-LLM vendor went from "
                 f"{ls['share'][yi[ly - 3]]:.0%} [{ls['ci95'][yi[ly - 3]][0]:.0%}–{ls['ci95'][yi[ly - 3]][1]:.0%}] of the total in {ly - 3} "
                 f"to {ls['share'][yi[ly]]:.0%} [{ls['ci95'][yi[ly]][0]:.0%}–{ls['ci95'][yi[ly]][1]:.0%}] in {ly}.")

    L.append("\n### Emerging signals (last 90 days)\n")
    L.append("![Emerging signals](docs/figures/emerging_signals.png)\n")
    L.append(f"Window {G['window'][0]} → {G['window'][1]} ({G['n_recent']} day-dated incidents) vs baseline {G['baseline'][0]} → {G['baseline'][1]} ({G['n_baseline']}). "
             f"Recent incidents by source: {G['recent_by_source']}.\n")
    if G["flagged"]:
        L.append("| Dimension | Category | Recent | Prior year | Share ratio | z | Example |")
        L.append("|---|---|---:|---:|---:|---:|---|")
        for s in G["flagged"][:12]:
            ex = s["examples"][0] if s["examples"] else None
            exs = f"[{ex['title'][:70]}]({ex['url']})" if ex else ""
            L.append(f"| {s['dimension'].replace('_', ' ')} | {s['label'].replace('_', ' ')} | {s['recent']} | {s['baseline']} | ×{s['share_ratio']} | {s['z']} | {exs} |")
    else:
        L.append("No category passed the flag threshold in this run.")
    L.append(f"\n{len(G['signals'])} categories were tested, so at z ≥ 2 about {max(1, round(len(G['signals']) * 0.023))} flags are expected by chance alone. Treat flags as leads for an analyst.")
    headline = {
        "n_incidents": S["n_incidents"], "aiaaic_merged": dd["aiaaic_merged_into_aiid"],
        "mit_domain_kappa": m["domain_7class"]["kappa"]["value"],
        "cset_sector_kappa": c["sector_coarse"]["kappa"]["value"],
        "cset_sector_hh": hh.get("sector_coarse", {}).get("weighted_mean_kappa"),
        "sev_auc_deaths": c["severity_auc_lives_lost"]["value"],
        "flagged": [f"{s['dimension']}:{s['label']} x{s['share_ratio']} (z={s['z']})" for s in G["flagged"][:5]],
        "complete": complete, "coverage": {s: f"{labelled.get(s, 0)}/{hn_eligible if s == 'hn' else n}" for s, n in total.items()},
        "generated": S["generated_at"],
    }
    return "\n".join(L), headline


def main():
    md, h = results_md()
    readme = ROOT / "README.md"
    txt = readme.read_text()
    block = f"<!-- RESULTS:START -->\n{md}\n<!-- RESULTS:END -->"
    if "<!-- RESULTS:START -->" in txt:
        txt = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda _: block, txt, flags=re.S)
    else:
        txt = txt.replace("{{RESULTS}}", block)
    readme.write_text(txt)
    status = f"""# STATUS: ai-incident-radar

- **Repo:** {REPO}
- **Dashboard (GitHub Pages):** {PAGES}
- **Last pipeline run:** {h['generated']}
- **Run complete:** {"yes" if h['complete'] else "NO, partial (Gemini free-tier daily quota); coverage " + str(h['coverage'])}

## Headline
- {h['n_incidents']:,} canonical AI incidents from AIID + AIAAIC + Hacker News; {h['aiaaic_merged']} AIAAIC entries deduplicated into AIID incidents (embedding retrieval + LLM-judge verification).
- Gemini classifier vs human CSET labels: sector κ = {h['cset_sector_kappa']:.2f} (CSET annotator-vs-annotator κ = {h['cset_sector_hh']:.2f}); severity AUROC for fatal incidents = {h['sev_auc_deaths']:.2f}. MIT risk-domain κ = {h['mit_domain_kappa']:.2f} (vs LLM-generated MIT labels).
- Emerging signals (last 90 days): {"; ".join(h['flagged']) or "none flagged"}.

## Left undone / caveats
- Reddit public JSON returns HTTP 403 without OAuth, so Reddit is skipped (handled gracefully; the code path stays in place).
- No Hugging Face dataset upload (no HF token).
- Free-tier Gemini quota: 500 requests/day per flash-lite model and 20/day per flash model, shared across six projects. Classification is cached and incremental; the weekly Action and `run_all.sh` resume where the last run stopped.
- Weekly refresh: `.github/workflows/refresh.yml` (Mondays 06:17 UTC, plus manual dispatch) with the `GEMINI_API_KEY` repo secret.
"""
    (ROOT / "STATUS.md").write_text(status)
    print(status)


if __name__ == "__main__":
    main()
