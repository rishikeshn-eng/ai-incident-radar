# AI Incident Radar

**An open, weekly-refreshed AI incident intelligence pipeline: ingest → deduplicate → classify → validate → detect emerging signals.**

**Dashboard:** https://rishikeshn-eng.github.io/ai-incident-radar/ · **Data:** [`docs/data/`](docs/data) · **Validation:** [`results/validation.json`](results/validation.json)

![Incidents by MIT risk domain](docs/figures/incidents_by_domain.png)

## Problem

When an aircraft crashes, an independent investigator (NTSB, AAIB, BEA) collects evidence, classifies root causes into a shared taxonomy, and publishes trends the whole industry learns from. AI has nothing comparable. Evidence about real-world AI failures is scattered across volunteer databases, journalism, and forum posts, each with its own schema, coverage and lag. Coefficient Giving's [Project Tailwind](https://80000hours.org/podcast/episodes/max-nadeau-project-tailwind-technical-ai-grants/) lists third-party evidence generation and incident detection among the AI safety problems that "no one currently owns".

Concretely:
- The two largest public collections, the [AI Incident Database](https://incidentdatabase.ai) (AIID) and the [AIAAIC Repository](https://www.aiaaic.org/aiaaic-repository), overlap heavily but share no IDs. Counting either one alone double-counts nothing and misses a lot; naively adding them double-counts a lot.
- Human taxonomy coverage is thin. CSET's AI Harm Taxonomy v1, the most detailed human annotation, covers about 200 published AIID incidents out of 1,751.
- Curated databases lag. AIID incidents dated in the last 90 days are a fraction of the yearly rate, so any "what is new" view needs faster community signals (Hacker News, Reddit), which are noisy and mostly not incidents.

This project builds the missing layer: a reproducible pipeline that merges these sources, labels every incident in one safety-oriented taxonomy, measures how far those labels can be trusted, and flags categories that are rising.

## What it does

| Stage | Module | What happens |
|---|---|---|
| 1. Ingest | [`radar/ingest.py`](radar/ingest.py) | Latest AIID full snapshot (incidents + reports + taxonomy CSVs), the AIAAIC "Incidents" sheet, Hacker News stories via the Algolia API (40 incident-oriented queries, ≥20 points, last 455 days), Reddit public JSON (skipped if blocked, which it currently is: HTTP 403). |
| 2. Dedupe + cluster | [`radar/dedupe.py`](radar/dedupe.py) | `bge-base-en-v1.5` embeddings (local ONNX via fastembed, cached). Union-find merge: AIAAIC→AIID at cosine ≥ T_INC with year ±1; HN→known incident at cosine ≥ T_NEWS; HN↔HN reposts within 14 days. HDBSCAN topic clusters on a 32-d PCA projection, named with c-TF-IDF. |
| 3. Classify | [`radar/classify.py`](radar/classify.py) | Gemini (structured JSON output, temperature 0, 20 records per call) assigns: MIT AI Risk Repository subdomain (24) → domain (7), MIT entity / intent / timing, multi-label **failure modes** (hallucination, sycophancy, agent misuse, prompt injection, deepfake, bias, privacy, autonomy / loss-of-control signals, and 9 more), severity 1–5, harm types, sector, model family, and an `is_ai_incident` filter for community posts. Every response is cached in [`labels/`](labels), so reruns only pay for new incidents. |
| 4. Validate | [`radar/validate.py`](radar/validate.py) | Cohen's κ, accuracy and macro-F1 with 1,000-sample bootstrap CIs against three independent label sets (below), plus a human–human baseline from CSET's individual annotator files and an LLM-judge audit of the dedupe thresholds. |
| 5. Trends + signals | [`radar/trends.py`](radar/trends.py) | Yearly counts and shares by MIT domain, failure mode, sector, model family and harm type; mean severity with CIs; LLM-vendor share with Wilson CIs. **Emerging signals:** share of each category among day-dated incidents in the last 90 days vs the previous 365 days, as a log share-ratio z-test. |
| 6. Publish | [`docs/`](docs), [`.github/workflows/refresh.yml`](.github/workflows/refresh.yml) | Static GitHub Pages dashboard (search, 7 filters, CSV export, charts, signal cards, validation table). A weekly GitHub Action reruns everything with the `GEMINI_API_KEY` repo secret and commits the refreshed data. |

{{RESULTS}}

## Related work

- **AI Incident Database** (McGregor, 2021, [arXiv:2011.08512](https://arxiv.org/abs/2011.08512)): the main curated collection, with several taxonomies attached: CSET v0/v1 (human), GMF failure modes, and MIT risk domains.
- **AIAAIC Repository**: independent, journalist-led, broader "incidents and controversies" scope, with editorial tags for technology, ethical issue and harm.
- **MIT AI Risk Repository** (Slattery et al., 2024, [arXiv:2408.12622](https://arxiv.org/abs/2408.12622)): the causal and domain taxonomy used here. The **MIT AI Incident Tracker** ([AIID blog](https://incidentdatabase.ai/blog/scalable-ai-incident-classification/)) classifies AIID incidents into it with an LLM (Claude Sonnet) and reports about 80% consistency with expert ratings. Those are the "MIT" labels in the AIID snapshot, which matters for interpreting the validation below.
- **CSET AI Harm Taxonomy** ([Hoffmann & Frase, 2023](https://cset.georgetown.edu/publication/adding-structure-to-ai-harm/)): human, peer-reviewed annotations of harm, sector, intent and autonomy.
- **OECD AI Incidents Monitor** ([oecd.ai/incidents](https://oecd.ai/en/incidents)): automated news-based incident detection at scale, with a different taxonomy and no public validation against human labels.
- **Incident analysis for AI agents** ([arXiv:2508.14231](https://arxiv.org/abs/2508.14231)): argues incident reports need agent-specific fields, which motivates the `agent_misuse`, `prompt_injection` and `autonomy_loss_of_control` failure modes here.

What this adds: (1) a cross-database dedupe, so counts are not inflated by the AIID/AIAAIC overlap; (2) safety-specific failure modes (sycophancy, agent misuse, prompt injection, loss-of-control signals) that existing taxonomies do not tag directly; (3) a classifier validated against human labels, compared with human–human agreement on the same items; (4) a statistical emerging-signals detector that uses fast community sources, refreshed weekly.

## Limitations

- **Reporting bias.** Every count here is a count of *reported and curated* incidents. Media attention, the AIID/AIAAIC editorial focus (English-language, US/EU-centric), and curation lag all shape the trends. Shares are more robust than counts, but not immune.
- **Labels are model-generated.** κ against humans is moderate at best for most fields (see validation). Use per-incident labels as triage, not ground truth. The MIT reference is itself LLM-generated, so agreement with it measures consistency with another model, not correctness.
- **Severity is a coarse 1–5 judgment** from short text. It ranks incidents sensibly against CSET's human harm levels, but it is not calibrated to real-world harm magnitude.
- **Community sources:** Reddit blocks unauthenticated JSON (HTTP 403), so it is currently skipped. HN coverage is limited to 40 search queries at ≥20 points, which biases the signal toward tech-industry stories (coding agents, LLM products).
- **Dedupe** uses one embedding threshold per source pair. The LLM-judge audit estimates its precision, but some distinct incidents with near-identical wording (e.g. repeated lawyer-hallucination cases) can merge, and paraphrased duplicates can survive.
- **Emerging signals** are a screening statistic (z ≥ 2 across ~100 tested categories means about 2–3 false flags expected by chance). They are leads for a human analyst, not findings.
- Free-tier Gemini quotas (15 requests/min shared across projects) forced batching 20 records per call. Batch context could slightly bias labels compared with one call per record. This was not measured.
- No Hugging Face upload: no HF token was available.

## Reproduce

```bash
git clone https://github.com/rishikeshn-eng/ai-incident-radar && cd ai-incident-radar
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python -r requirements.txt
export GEMINI_API_KEY=...        # free tier works; cached labels in labels/ make reruns cheap
./run_all.sh                     # ingest -> dedupe -> classify -> validate -> trends -> figures
python -m http.server -d docs    # open http://localhost:8000
```

Individual stages: `python -m radar.<ingest|dedupe|classify|validate|trends|figures>`. `python -m radar.validate --dedupe-audit` also runs the LLM-judge audit of dedupe thresholds.

## Data licences

AIID snapshot: CC BY-SA 4.0. AIAAIC Repository: CC BY-SA 4.0 (attribution: AIAAIC). Hacker News content via the public Algolia API. Raw datasets are not redistributed in this repo, only derived labels and aggregates.
