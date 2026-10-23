# STATUS: ai-incident-radar

- **Repo:** https://github.com/rishikeshn-eng/ai-incident-radar
- **Dashboard (GitHub Pages):** https://rishikeshn-eng.github.io/ai-incident-radar/
- **Last pipeline run:** 2026-10-23T22:11:10+00:00
- **Run complete:** NO, partial (Gemini free-tier daily quota); coverage {'aiaaic': '1008/2259', 'aiid': '1747/1751', 'hn': '0/705'}

## Headline
- 2,515 canonical AI incidents from AIID + AIAAIC + Hacker News; 421 AIAAIC entries deduplicated into AIID incidents (embedding retrieval + LLM-judge verification).
- Gemini classifier vs human CSET labels: sector κ = 0.66 (CSET annotator-vs-annotator κ = 0.65); severity AUROC for fatal incidents = 0.95. MIT risk-domain κ = 0.64 (vs LLM-generated MIT labels).
- Emerging signals (last 90 days): cluster:claude / opus / production / anthropic x2.86 (z=4.1); model_family:anthropic x3.51 (z=3.37); failure_mode:agent_misuse x2.13 (z=2.86); mit_subdomain:2.2 AI system security vulnerabilities and attacks x3.29 (z=2.62); failure_mode:cyberattack x2.93 (z=2.61).

## Left undone / caveats
- Reddit public JSON returns HTTP 403 without OAuth, so Reddit is skipped (handled gracefully; the code path stays in place).
- No Hugging Face dataset upload (no HF token).
- Free-tier Gemini quota: 500 requests/day per flash-lite model and 20/day per flash model, shared across six projects. Classification is cached and incremental; the weekly Action and `run_all.sh` resume where the last run stopped.
- Weekly refresh: `.github/workflows/refresh.yml` (Mondays 06:17 UTC, plus manual dispatch) with the `GEMINI_API_KEY` repo secret.
