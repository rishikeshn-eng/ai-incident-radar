"""Stage 1: ingest AIID snapshot, AIAAIC repository, and community reports (HN, Reddit).

Writes data/processed/records.parquet: one row per source record with a common schema:
  uid, source, source_id, title, text, date, year, url, deployer, developer, extra (json)
"""
from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from .common import AIAAIC_INCIDENTS_GID, AIAAIC_SHEET, AIID_SNAPSHOTS, PROC, RAW, UA

csv.field_size_limit(10**9)


# ---------------------------------------------------------------- AIID
def fetch_aiid() -> "Path":
    """Download (if needed) the newest AIID full snapshot and return the extracted CSV dir."""
    html = requests.get(AIID_SNAPSHOTS, headers={"User-Agent": UA}, timeout=60).text
    urls = sorted(set(re.findall(r"https://[^\"']+backup-\d+\.tar\.bz2", html)))
    if not urls:
        raise RuntimeError("no AIID snapshot links found")
    url = urls[-1]
    tag = re.search(r"backup-(\d+)", url).group(1)
    dest = RAW / "aiid" / tag
    csv_dir = dest / "mongodump_full_snapshot"
    if (csv_dir / "incidents.csv").exists():
        return csv_dir
    dest.mkdir(parents=True, exist_ok=True)
    tar = RAW / f"aiid-{tag}.tar.bz2"
    # curl with resume: the R2 bucket occasionally stalls mid-transfer.
    for _ in range(8):
        r = subprocess.run(["curl", "-sSL", "--speed-limit", "20000", "--speed-time", "30",
                            "-C", "-", "-o", str(tar), url])
        if r.returncode == 0:
            break
    subprocess.run(["tar", "-xjf", str(tar), "-C", str(dest),
                    "mongodump_full_snapshot/incidents.csv",
                    "mongodump_full_snapshot/reports.csv",
                    "mongodump_full_snapshot/classifications_MIT.csv",
                    "mongodump_full_snapshot/classifications_CSETv1.csv",
                    "mongodump_full_snapshot/classifications_CSETv1_Annotator-1.csv",
                    "mongodump_full_snapshot/classifications_CSETv1_Annotator-2.csv",
                    "mongodump_full_snapshot/classifications_CSETv1_Annotator-3.csv",
                    "mongodump_full_snapshot/classifications_GMF.csv"], check=True)
    tar.unlink(missing_ok=True)
    return csv_dir


def _jlist(s: str) -> list[str]:
    try:
        v = json.loads(s)
        return [str(x) for x in v] if isinstance(v, list) else []
    except Exception:
        return []


def load_aiid(csv_dir) -> pd.DataFrame:
    inc = pd.read_csv(csv_dir / "incidents.csv", dtype=str).fillna("")
    rep = pd.read_csv(csv_dir / "reports.csv", dtype=str, engine="python").fillna("")
    rep = rep.set_index("report_number")
    rows = []
    for r in inc.itertuples(index=False):
        rep_ids = [str(x) for x in _jlist(r.reports)]
        reps = rep.loc[rep.index.intersection(rep_ids)]
        # Lead paragraph of the earliest English report gives the classifier more context.
        lead = ""
        if len(reps):
            en = reps[reps["language"].isin(["en", ""])]
            pick = (en if len(en) else reps).sort_values("date_published").iloc[0]
            lead = re.sub(r"\s+", " ", pick["text"])[:1200]
        domains = sorted(set(reps["source_domain"])) if len(reps) else []
        rows.append(dict(
            uid=f"aiid:{r.incident_id}", source="aiid", source_id=r.incident_id,
            title=r.title, text=f"{r.description}\n\nReport excerpt: {lead}".strip(),
            date=r.date, url=f"https://incidentdatabase.ai/cite/{r.incident_id}",
            deployer=", ".join(_jlist(r[4])), developer=", ".join(_jlist(r[5])),
            extra=json.dumps({"harmed": _jlist(r[6]), "n_reports": len(rep_ids),
                              "domains": domains[:12]}),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- AIAAIC
AIAAIC_COLS = ["id", "headline", "occurred", "deployer", "developer", "system", "technology",
               "purpose", "news_trigger", "ethical_issue", "jurisdiction", "sector",
               "harm_individual", "harm_societal", "harm_environmental", "harm_status",
               "consequence", "response", "link"]


def load_aiaaic() -> pd.DataFrame:
    url = (f"https://docs.google.com/spreadsheets/d/{AIAAIC_SHEET}/export"
           f"?format=csv&gid={AIAAIC_INCIDENTS_GID}")
    path = RAW / "aiaaic" / "incidents.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    except Exception as e:  # fall back to the last good copy
        print(f"[aiaaic] download failed ({e}); using cached copy", file=sys.stderr)
    rows = list(csv.reader(io.open(path, encoding="utf-8")))
    # The sheet has a title row plus a two-row header; data rows start with "AIAAIC".
    out = []
    for row in rows:
        if not row or not row[0].startswith("AIAAIC"):
            continue
        d = dict(zip(AIAAIC_COLS, (c.strip() for c in row)))
        yr = re.search(r"(19|20)\d\d", d.get("occurred", ""))
        harms = "; ".join(x for x in (d.get("harm_individual"), d.get("harm_societal"),
                                      d.get("harm_environmental")) if x)
        text = (f"{d['headline']}. System: {d.get('system','')}. Technology: {d.get('technology','')}. "
                f"Purpose: {d.get('purpose','')}. Ethical issues: {d.get('ethical_issue','')}. "
                f"Harms: {harms}. Sector: {d.get('sector','')}. Where: {d.get('jurisdiction','')}.")
        out.append(dict(
            uid=f"aiaaic:{d['id']}", source="aiaaic", source_id=d["id"], title=d["headline"],
            text=text, date=f"{yr.group(0)}-07-01" if yr else "", url=d.get("link", ""),
            deployer=d.get("deployer", ""), developer=d.get("developer", ""),
            extra=json.dumps({k: d.get(k, "") for k in
                              ("technology", "ethical_issue", "sector", "jurisdiction",
                               "harm_status", "consequence", "news_trigger", "system")}),
        ))
    df = pd.DataFrame(out).drop_duplicates("uid")
    df["date_precision"] = "year"
    return df


# ---------------------------------------------------------------- community
HN_QUERIES = [
    "AI hallucination lawsuit", "chatbot hallucinated", "AI agent deleted", "prompt injection",
    "jailbreak LLM", "deepfake scam", "AI deepfake", "AI bias lawsuit", "facial recognition wrongful",
    "ChatGPT lawsuit", "AI chatbot suicide", "sycophantic", "sycophancy", "AI agent incident",
    "Copilot leaked", "data leak LLM", "AI privacy breach", "Grok", "self-driving crash",
    "Tesla Autopilot crash", "Waymo crash", "AI generated misinformation", "voice clone fraud",
    "AI safety incident", "model refused shutdown", "reward hacking", "AI agent wiped database",
    "AI coding agent", "LLM exfiltration", "AI fabricated citations", "AI nudify",
    "AI generated CSAM", "algorithm wrongly denied", "AI mistake", "Gemini AI",
    "Claude agent", "OpenAI model", "AI scam", "AI misuse", "MCP vulnerability",
]


def load_hn(days: int = 455) -> pd.DataFrame:
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    seen, out = set(), []
    for q in HN_QUERIES:
        for page in range(3):
            try:
                r = requests.get("https://hn.algolia.com/api/v1/search",
                                 params={"query": q, "tags": "story", "hitsPerPage": 100,
                                         "page": page,
                                         "numericFilters": f"created_at_i>{since},points>=20"},
                                 headers={"User-Agent": UA}, timeout=30)
                r.raise_for_status()
                hits = r.json().get("hits", [])
            except Exception as e:
                print(f"[hn] {q!r} p{page}: {e}", file=sys.stderr)
                break
            for h in hits:
                oid = h["objectID"]
                if oid in seen or not h.get("title"):
                    continue
                seen.add(oid)
                out.append(dict(
                    uid=f"hn:{oid}", source="hn", source_id=oid, title=h["title"],
                    text=f"{h['title']}. {re.sub('<[^>]+>', ' ', h.get('story_text') or '')[:800]}",
                    date=h["created_at"][:10],
                    url=h.get("url") or f"https://news.ycombinator.com/item?id={oid}",
                    deployer="", developer="",
                    extra=json.dumps({"points": h.get("points", 0),
                                      "comments": h.get("num_comments", 0),
                                      "hn": f"https://news.ycombinator.com/item?id={oid}",
                                      "query": q}),
                ))
            if len(hits) < 100:
                break
            time.sleep(0.3)
    df = pd.DataFrame(out)
    df["date_precision"] = "day"
    return df


REDDIT_SUBS = ["artificial", "ChatGPT", "OpenAI", "LocalLLaMA", "singularity", "ClaudeAI"]
REDDIT_QUERIES = ["incident", "hallucinated", "deleted my", "prompt injection", "jailbreak",
                  "deepfake", "lawsuit", "leaked", "sycophantic"]


def load_reddit(days: int = 455) -> pd.DataFrame:
    """Public JSON search. Reddit frequently 403s unauthenticated clients; skip gracefully."""
    since = time.time() - days * 86400
    out = []
    for sub in REDDIT_SUBS:
        for q in REDDIT_QUERIES:
            try:
                r = requests.get(f"https://www.reddit.com/r/{sub}/search.json",
                                 params={"q": q, "restrict_sr": 1, "sort": "top", "t": "year",
                                         "limit": 50},
                                 headers={"User-Agent": UA}, timeout=20)
            except Exception as e:
                print(f"[reddit] unreachable ({e}); skipping", file=sys.stderr)
                return pd.DataFrame()
            if r.status_code != 200:
                print(f"[reddit] HTTP {r.status_code}; skipping Reddit source", file=sys.stderr)
                return pd.DataFrame()
            for c in r.json().get("data", {}).get("children", []):
                d = c["data"]
                if d.get("created_utc", 0) < since or d.get("score", 0) < 50:
                    continue
                out.append(dict(
                    uid=f"reddit:{d['id']}", source="reddit", source_id=d["id"], title=d["title"],
                    text=f"{d['title']}. {(d.get('selftext') or '')[:800]}",
                    date=datetime.fromtimestamp(d["created_utc"], timezone.utc).strftime("%Y-%m-%d"),
                    url="https://www.reddit.com" + d["permalink"], deployer="", developer="",
                    extra=json.dumps({"score": d.get("score"), "sub": sub}),
                ))
            time.sleep(1.0)
    df = pd.DataFrame(out).drop_duplicates("uid") if out else pd.DataFrame()
    if len(df):
        df["date_precision"] = "day"
    return df


def main():
    csv_dir = fetch_aiid()
    aiid = load_aiid(csv_dir)
    aiid["date_precision"] = "day"
    print(f"[aiid] {len(aiid)} incidents from {csv_dir.parent.name}")
    aiaaic = load_aiaaic()
    print(f"[aiaaic] {len(aiaaic)} incidents")
    hn = load_hn()
    print(f"[hn] {len(hn)} stories")
    rd = load_reddit()
    print(f"[reddit] {len(rd)} posts")
    df = pd.concat([aiid, aiaaic, hn, rd], ignore_index=True).fillna("")
    df["year"] = pd.to_numeric(df["date"].str[:4], errors="coerce")
    df.to_parquet(PROC / "records.parquet", index=False)
    meta = {"aiid_snapshot": csv_dir.parent.name, "aiid_csv_dir": str(csv_dir),
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "counts": df["source"].value_counts().to_dict(),
            "reddit_status": "ok" if len(rd) else "skipped (HTTP 403 / unreachable)"}
    (PROC / "ingest_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
