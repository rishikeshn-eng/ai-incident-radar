#!/usr/bin/env bash
# Full refresh: ingest -> dedupe/cluster -> classify -> validate -> trends/dashboard data -> figures -> README/STATUS.
# Every Gemini response is cached under labels/, so reruns only pay for new incidents.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-.venv/bin/python}
[ -x "$PY" ] || PY=python
$PY -m radar.ingest
$PY -m radar.dedupe
$PY -m radar.classify
$PY -m radar.validate ${VALIDATE_ARGS:-}
$PY -m radar.trends
$PY -m radar.figures
$PY -m radar.report
