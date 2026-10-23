"""Stage 3: classify each canonical incident with Gemini into the radar taxonomy.

Batched (several incidents per request), structured JSON output, 4 concurrent requests max,
exponential backoff on 429/5xx, and every response cached to labels/<model>.jsonl so reruns
(and the weekly GitHub Action) only pay for new or edited incidents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .common import LABELS, PROC, read_jsonl
from .taxonomy import (FAILURE_MODES, HARM_TYPES, MIT_SUBDOMAINS, MODEL_FAMILIES, SECTORS,
                       SEVERITY_GUIDE)

PROMPT_VERSION = "v1"
DEFAULT_MODEL = "gemini-3.1-flash-lite"
BATCH = 20
# Free-tier key shared by several projects: ~15 RPM per model in total. Self-limit hard.
MIN_INTERVAL = float(os.environ.get("RADAR_MIN_INTERVAL", "10"))
_last_call = [0.0]
_rate_lock = threading.Lock()


def _wait_turn():
    with _rate_lock:
        now = time.time()
        wait = _last_call[0] + MIN_INTERVAL - now
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()

SYSTEM = f"""You are an AI incident analyst building a dataset like those kept by aviation accident investigators.
For each record you receive, classify it. Use only the record text; do not invent facts.

Fields:
- is_ai_incident: true if the record describes a concrete event (or credible near-miss / demonstrated vulnerability) where an AI or algorithmic system caused, contributed to, or enabled harm. False for product announcements, opinion pieces, tutorials, funding news, or general discussion.
- mit_subdomain: one code from the MIT AI Risk Repository causal-domain taxonomy (pick the single best fit):
{chr(10).join(f'  {k} {v}' for k, v in MIT_SUBDOMAINS.items())}
- entity: "AI" if the harm stems mainly from a decision/action/output of the AI system itself; "Human" if a human used AI to cause it (misuse, intentional); "Other" if ambiguous.
- intent: "Intentional" if the harm was an expected outcome of someone's goal; "Unintentional" if unexpected; "Other" if ambiguous.
- timing: "Pre-deployment" (found in testing/research/red-teaming before release), "Post-deployment", or "Other".
- failure_modes: list of zero or more of:
{chr(10).join(f'  {k}: {v}' for k, v in FAILURE_MODES.items())}
- primary_failure_mode: the single most important item from failure_modes (or "robustness_error").
- severity (1-5):
{SEVERITY_GUIDE}
- harm_types: list from {HARM_TYPES}
- sector: one of {SECTORS}
- model_family: one of {MODEL_FAMILIES} (the developer/model family of the AI system involved)
- system_name: short name of the AI system/product if stated, else ""
- summary: one neutral sentence (<= 30 words).
Return a JSON array with one object per record, in the same order, each including its "uid"."""

SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "uid": {"type": "STRING"},
            "is_ai_incident": {"type": "BOOLEAN"},
            "mit_subdomain": {"type": "STRING", "enum": list(MIT_SUBDOMAINS)},
            "entity": {"type": "STRING", "enum": ["AI", "Human", "Other"]},
            "intent": {"type": "STRING", "enum": ["Intentional", "Unintentional", "Other"]},
            "timing": {"type": "STRING", "enum": ["Pre-deployment", "Post-deployment", "Other"]},
            "failure_modes": {"type": "ARRAY", "items": {"type": "STRING", "enum": list(FAILURE_MODES)}},
            "primary_failure_mode": {"type": "STRING", "enum": list(FAILURE_MODES)},
            "severity": {"type": "INTEGER"},
            "harm_types": {"type": "ARRAY", "items": {"type": "STRING", "enum": HARM_TYPES}},
            "sector": {"type": "STRING", "enum": SECTORS},
            "model_family": {"type": "STRING", "enum": MODEL_FAMILIES},
            "system_name": {"type": "STRING"},
            "summary": {"type": "STRING"},
        },
        "required": ["uid", "is_ai_incident", "mit_subdomain", "entity", "intent", "timing",
                     "failure_modes", "primary_failure_mode", "severity", "harm_types", "sector",
                     "model_family", "system_name", "summary"],
    },
}


def record_hash(title: str, text: str) -> str:
    return hashlib.sha1(f"{PROMPT_VERSION}|{title}|{text}".encode()).hexdigest()[:16]


def render(rec: dict) -> str:
    parts = [f"uid: {rec['uid']}", f"title: {rec['title']}", f"date: {rec['date']}"]
    if rec.get("deployer"):
        parts.append(f"deployer: {rec['deployer']}")
    if rec.get("developer"):
        parts.append(f"developer: {rec['developer']}")
    parts.append(f"text: {rec['text'][:1600]}")
    return "\n".join(parts)


class Classifier:
    def __init__(self, model: str):
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model
        self.cache_path = LABELS / f"{model}.jsonl"
        self.lock = threading.Lock()
        self.cache = {(r["uid"], r["_hash"]): r for r in read_jsonl(self.cache_path)}

    def _call(self, recs: list[dict]) -> list[dict]:
        prompt = "\n\n---\n\n".join(render(r) for r in recs)
        cfg = self.types.GenerateContentConfig(
            system_instruction=SYSTEM, temperature=0.0,
            response_mime_type="application/json", response_schema=SCHEMA,
        )
        delay = 4.0
        for attempt in range(8):
            _wait_turn()
            try:
                resp = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                out = json.loads(resp.text)
                if isinstance(out, list):
                    return out
            except Exception as e:  # 429 / 5xx / malformed JSON
                msg = str(e)
                if attempt == 7:
                    print(f"[classify] giving up on batch: {msg[:200]}", file=sys.stderr)
                    return []
                if "429" not in msg and "RESOURCE_EXHAUSTED" not in msg and "503" not in msg \
                        and "500" not in msg and "404" not in msg and "JSON" not in msg and "Expecting" not in msg:
                    print(f"[classify] error: {msg[:200]}", file=sys.stderr)
            time.sleep(delay + random.random() * 2)
            delay = min(delay * 2, 120)
        return []

    def run_batch(self, recs: list[dict]) -> int:
        out = self._call(recs)
        by_uid = {o.get("uid"): o for o in out if isinstance(o, dict)}
        # Fall back to positional matching if the model mangled uids.
        if len(by_uid) < len(recs) and len(out) == len(recs):
            by_uid = {r["uid"]: o for r, o in zip(recs, out)}
        n = 0
        with self.lock, open(self.cache_path, "a", encoding="utf-8") as f:
            for r in recs:
                o = by_uid.get(r["uid"])
                if not o:
                    continue
                o["uid"] = r["uid"]
                o["_hash"] = r["_hash"]
                o["_model"] = self.model
                o["severity"] = int(min(5, max(1, int(o.get("severity") or 1))))
                self.cache[(r["uid"], r["_hash"])] = o
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
                n += 1
        return n

    def classify(self, df: pd.DataFrame, workers: int = 4) -> pd.DataFrame:
        recs = df[["uid", "title", "text", "date", "deployer", "developer"]].to_dict("records")
        for r in recs:
            r["_hash"] = record_hash(r["title"], r["text"])
        todo = [r for r in recs if (r["uid"], r["_hash"]) not in self.cache]
        print(f"[classify:{self.model}] {len(recs)} records, {len(todo)} uncached", flush=True)
        batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
        done = 0
        with ThreadPoolExecutor(max_workers=min(workers, 4)) as ex:
            futs = [ex.submit(self.run_batch, b) for b in batches]
            for i, fu in enumerate(as_completed(futs), 1):
                done += fu.result()
                if i % 5 == 0 or i == len(futs):
                    print(f"  {i}/{len(futs)} batches, {done} labelled", flush=True)
        rows = [self.cache.get((r["uid"], r["_hash"])) for r in recs]
        return pd.DataFrame([r for r in rows if r])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--subset", choices=["all", "validation"], default="all",
                    help="validation = only AIID incidents (the ones with reference labels)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    # Classify source records (not merged groups) so labels are stable across dedupe changes;
    # each canonical incident inherits the label of its primary record.
    canon = pd.read_parquet(PROC / "records.parquet")
    if args.subset == "validation":
        canon = canon[canon["source"] == "aiid"]
    if args.limit:
        canon = canon.head(args.limit)
    labels = Classifier(args.model).classify(canon)
    out = PROC / f"labels_{args.model}.parquet"
    labels.drop(columns=[c for c in labels.columns if c.startswith("_")]).to_parquet(out, index=False)
    print(f"wrote {out} ({len(labels)} rows)")


if __name__ == "__main__":
    main()
