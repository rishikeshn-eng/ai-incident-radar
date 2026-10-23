"""Shared paths and helpers for the AI Incident Radar pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
# Model outputs are small, derived, and expensive to regenerate, so they are committed.
LABELS = ROOT / "labels"
RESULTS = ROOT / "results"
SITE = ROOT / "docs"

for p in (RAW, PROC, LABELS, RESULTS, SITE / "data"):
    p.mkdir(parents=True, exist_ok=True)

AIID_SNAPSHOTS = "https://incidentdatabase.ai/research/snapshots/"
AIAAIC_SHEET = "1Bn55B4xz21-_Rgdr8BBb2lt0n_4rzLGxFADMlVW0PYI"
AIAAIC_INCIDENTS_GID = "888071280"
UA = "ai-incident-radar/0.1 (+https://github.com/rishikeshn-eng/ai-incident-radar)"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # tolerate a torn last line from an interrupted run
    return out


def write_json(path: Path, obj, indent=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, default=str)
    os.replace(tmp, path)
