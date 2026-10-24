# STATUS: ai-incident-radar

- **Repo:** https://github.com/rishikeshn-eng/ai-incident-radar
- **Dashboard (GitHub Pages):** https://rishikeshn-eng.github.io/ai-incident-radar/
- **Last pipeline run:** 2026-10-24T10:02:07+00:00
- **Run complete:** yes

## Headline
- 3,637 canonical AI incidents from AIID + AIAAIC + Hacker News; 421 AIAAIC entries deduplicated into AIID incidents (embedding retrieval + LLM-judge verification).
- Gemini classifier vs human CSET labels: sector κ = 0.66 (CSET annotator-vs-annotator κ = 0.65); severity AUROC for fatal incidents = 0.95. MIT risk-domain κ = 0.64 (vs LLM-generated MIT labels).
- Emerging signals (last 90 days): cluster:production / deleted / claude opus / coding agents x2.56 (z=5.15); sector:technology_software x2.17 (z=4.4); mit_subdomain:2.1 Compromise of privacy by obtaining, leaking or correctly inferring sensitive information x2.88 (z=3.63); failure_mode:privacy x2.43 (z=3.37); failure_mode:agent_misuse x2.09 (z=3.26).

## Left undone / caveats
- Reddit public JSON returns HTTP 403 without OAuth, so Reddit is skipped (handled gracefully; the code path stays in place).
- No Hugging Face dataset upload (no HF token).
- Free-tier Gemini quota: 500 requests/day per flash-lite model and 20/day per flash model, shared across six projects. Classification is cached and incremental; the weekly Action and `run_all.sh` resume where the last run stopped.
- Weekly refresh: `.github/workflows/refresh.yml` (Mondays 06:17 UTC, plus manual dispatch) with the `GEMINI_API_KEY` repo secret.
