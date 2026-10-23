"""Stage 2: embed every record, merge cross-source duplicates, and cluster canonical incidents.

Embeddings: BAAI/bge-base-en-v1.5 via fastembed (ONNX, CPU, 768-d), cached per record
in data/processed/emb_cache.npz so weekly refreshes only embed new records.

Dedupe rule (retrieve, verify, union-find):
  * candidates: each record's top-2 neighbours with cosine >= T_CAND_* (AIAAIC->AIID with
    |year diff| <= 1; HN->any curated incident; HN<->HN within 30 days)
  * every candidate pair is verified by an LLM judge (cached); confirmed pairs are merged
Canonical record per group: AIID > AIAAIC > community (highest score).
validate.py --dedupe-audit estimates how many duplicates fall below the candidate threshold.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import LABELS, PROC, ROOT, read_jsonl, write_json

EMB_MODEL = "BAAI/bge-base-en-v1.5"
DIM = 768
TOPK = 2
N_TOPICS = 40
T_CAND_INC = 0.76     # candidate retrieval thresholds (bge-base cosine), verified by the judge
T_CAND_NEWS = 0.74
T_FALLBACK = 0.90    # used only when no judge verdict is available
CACHE = PROC / "emb_cache.npz"


def emb_text(r) -> str:
    return f"{r['title']}. {r['text'][:500]}"


def _key(s: str) -> str:
    return hashlib.sha1(f"{EMB_MODEL}|{DIM}|{s}".encode()).hexdigest()[:20]


def embed(texts: list[str]) -> np.ndarray:
    """Local CPU ONNX embeddings (no torch, no API quota), cached per text."""
    from fastembed import TextEmbedding
    cache = {}
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        cache = dict(zip(z["keys"].tolist(), z["vecs"]))
    keys = [_key(t) for t in texts]
    todo = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in cache]
    print(f"[embed] {len(texts)} texts, {len(todo)} uncached", flush=True)
    if todo:
        model = TextEmbedding(EMB_MODEL, cache_dir=str(ROOT / "cache" / "fastembed"), threads=4)
        for i in range(0, len(todo), 512):
            chunk = todo[i:i + 512]
            for (k, _), v in zip(chunk, model.embed([t for _, t in chunk], batch_size=32)):
                v = np.asarray(v, dtype=np.float32)
                cache[k] = v / (np.linalg.norm(v) + 1e-9)
            _save(cache)
            print(f"  embedded {i + len(chunk)}/{len(todo)}", flush=True)
    return np.stack([cache[k] for k in keys])


def _save(cache: dict):
    ks = list(cache)
    np.savez(CACHE, keys=np.array(ks), vecs=np.stack([cache[k] for k in ks]))


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


JUDGE_MODEL = "gemini-3.1-flash-lite"
JUDGE_CACHE = LABELS / f"dedupe_judge_{JUDGE_MODEL}.jsonl"
JUDGE_BATCH = 40
JUDGE_PROMPT = """You are deduplicating AI incident databases. For each numbered pair, decide whether
A and B describe the SAME real-world incident: the same event, involving the same system/organisation
and the same harmed party. A news story or follow-up (lawsuit, fine, investigation) about that same
event counts as the same incident. Two different events of the same kind (e.g. two different robot
accidents, two different deepfake scams, two lawyers citing fake cases) are NOT the same.
Return a JSON array of objects {"pair": <number>, "same": true|false}, one per pair."""


def _rec_str(r) -> str:
    return f"{r['title']} ({str(r['date'])[:10]}). {r['text'][:350]}"


def judge_pairs(df: pd.DataFrame, cands: list[tuple[int, int]]) -> dict[tuple[str, str], bool]:
    """LLM verification of candidate duplicate pairs, cached by (uid_a, uid_b)."""
    cache = {(r["a"], r["b"]): r["same"] for r in read_jsonl(JUDGE_CACHE)}
    uids = df["uid"].to_numpy()
    todo = [(i, j) for i, j in cands if (uids[i], uids[j]) not in cache]
    print(f"[judge] {len(cands)} candidate pairs, {len(todo)} uncached", flush=True)
    if todo and os.environ.get("GEMINI_API_KEY"):
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        recs = df.to_dict("records")

        def run(batch):
            prompt = JUDGE_PROMPT + "\n\n" + "\n\n".join(
                f"Pair {k}:\nA: {_rec_str(recs[i])}\nB: {_rec_str(recs[j])}" for k, (i, j) in enumerate(batch))
            delay = 5
            for attempt in range(6):
                time.sleep(8 + random.random() * 4)  # keep well under the shared free-tier RPM
                try:
                    r = client.models.generate_content(
                        model=JUDGE_MODEL, contents=prompt,
                        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"))
                    out = {int(o["pair"]): bool(o["same"]) for o in json.loads(r.text) if "pair" in o}
                    return [(batch[k], out[k]) for k in range(len(batch)) if k in out]
                except Exception as e:
                    if "PerDay" in str(e):  # daily free-tier cap: unjudged pairs fall back to T_FALLBACK
                        print("[judge] daily quota exhausted; skipping", file=sys.stderr, flush=True)
                        return []
                    print(f"[judge] retry {attempt}: {str(e)[:120]}", file=sys.stderr, flush=True)
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
            return []

        batches = [todo[k:k + JUDGE_BATCH] for k in range(0, len(todo), JUDGE_BATCH)]
        with ThreadPoolExecutor(2) as ex, open(JUDGE_CACHE, "a") as f:
            for n, res in enumerate(ex.map(run, batches), 1):
                for (i, j), same in res:
                    cache[(uids[i], uids[j])] = same
                    f.write(json.dumps({"a": uids[i], "b": uids[j], "same": same}) + "\n")
                f.flush()
                if n % 10 == 0:
                    print(f"  judged {n}/{len(batches)} batches", flush=True)
    return cache


def dedupe(df: pd.DataFrame, E: np.ndarray):
    """Retrieve candidates by embedding similarity, confirm with an LLM judge, merge via union-find.

    Without an API key (or for pairs the judge never answered) fall back to a strict cosine
    threshold T_FALLBACK.
    """
    n = len(df)
    src = df["source"].to_numpy()
    year = df["year"].to_numpy(dtype=float)
    dates = pd.to_datetime(df["date"], errors="coerce")
    is_news = np.isin(src, ["hn", "reddit"])
    pairs, cands = [], []  # pairs: (i, j, sim, kind, year_ok) for every best match (audit)

    def year_ok(i, j):
        return bool(np.isnan(year[i]) or np.isnan(year[j]) or abs(year[i] - year[j]) <= 1)

    def topk(rows, cols, kind, t_cand, k=TOPK, extra=lambda i, j: True):
        S = E[rows] @ E[cols].T
        for r, i in enumerate(rows):
            order = np.argsort(-S[r])[:k]
            for rank, c in enumerate(order):
                j, s = cols[c], float(S[r, c])
                if j == i:
                    continue
                ok = year_ok(i, j) and extra(i, j)
                if rank == 0:
                    pairs.append((i, j, s, kind, ok))
                if s >= t_cand and ok:
                    cands.append((i, j, s, kind))

    aiaaic, aiid = np.where(src == "aiaaic")[0], np.where(src == "aiid")[0]
    topk(aiaaic, aiid, "aiaaic-aiid", T_CAND_INC)
    news, inc = np.where(is_news)[0], np.where(~is_news)[0]
    if len(news):
        topk(news, inc, "news-incident", T_CAND_NEWS)
        topk(news, news, "news-news", T_CAND_NEWS,
             extra=lambda i, j: j > i and abs((dates[i] - dates[j]).days) <= 30)
    verdict = judge_pairs(df, [(i, j) for i, j, _, _ in cands])
    uids = df["uid"].to_numpy()
    uf = UF(n)
    decided = []
    for i, j, s, kind in cands:
        v = verdict.get((uids[i], uids[j]))
        same = bool(v) if v is not None else s >= T_FALLBACK
        decided.append(dict(a=uids[i], b=uids[j], sim=s, kind=kind, same=same, judged=v is not None))
        if same:
            uf.union(i, j)
    groups = np.array([uf.find(i) for i in range(n)])
    return groups, pairs, pd.DataFrame(decided)


def score(r) -> tuple:
    pri = {"aiid": 0, "aiaaic": 1, "hn": 2, "reddit": 3}[r["source"]]
    ex = json.loads(r["extra"] or "{}")
    return (pri, -(ex.get("points") or ex.get("score") or 0))


NAME_STOP = {"ai", "reportedly", "allegedly", "purportedly", "usa", "technology", "purpose",
             "hn", "hey", "show", "ask", "new", "says", "using", "used", "use", "accused",
             "generated", "ai-generated", "artificial", "intelligence", "model", "models", "tool",
             "tools", "app", "via", "just", "like", "people", "report", "study", "incident"}


def label_clusters(titles: list[str], labels: np.ndarray, k: int = 4) -> dict[int, str]:
    """c-TF-IDF style: concatenate titles per cluster, take the top distinctive terms."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    ids = sorted(set(labels) - {-1})
    docs = [" ".join(t for t, l in zip(titles, labels) if l == c) for c in ids]
    if not docs:
        return {}
    vec = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS | NAME_STOP), ngram_range=(1, 2),
                          min_df=1, max_df=0.4, sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b")
    X = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for row, c in enumerate(ids):
        picked, words = [], set()
        for t in vocab[np.argsort(-X[row].toarray()[0])][:k * 4]:
            w = set(t.split())
            if w & words:  # skip terms that repeat a word already shown
                continue
            picked.append(t)
            words |= w
            if len(picked) == k:
                break
        out[c] = " / ".join(picked)
    return out


def cluster_topics(E: np.ndarray, titles: list[str], k: int = N_TOPICS):
    cl = KMeans(k, n_init=4, random_state=0).fit_predict(E)
    names = label_clusters(titles, cl)
    return cl, [names.get(c, "unclustered") for c in cl]


def main():
    df = pd.read_parquet(PROC / "records.parquet").reset_index(drop=True)
    E = embed([emb_text(r) for r in df.to_dict("records")])
    groups, pairs, decided = dedupe(df, E)
    decided.to_parquet(PROC / "dedupe_decisions.parquet", index=False)
    df["group"] = groups
    pd.DataFrame([dict(a=df.uid[i], b=df.uid[j], sim=s, kind=k, year_ok=y)
                  for i, j, s, k, y in pairs]).to_parquet(PROC / "dedupe_pairs.parquet", index=False)

    canon_rows = []
    for g, sub in df.groupby("group"):
        recs = sorted(sub.to_dict("records"), key=score)
        c = dict(recs[0])
        c["members"] = [r["uid"] for r in recs]
        c["n_sources"] = int(sub["source"].nunique())
        c["sources"] = sorted(sub["source"].unique().tolist())
        news = sub[sub["source"].isin(["hn", "reddit"])]
        c["community_mentions"] = int(len(news))
        c["community_points"] = int(sum(json.loads(x or "{}").get("points") or
                                        json.loads(x or "{}").get("score") or 0
                                        for x in news["extra"]))
        c["last_seen"] = max(sub["date"].replace("", np.nan).dropna(), default="")
        # Extra context for the classifier when an AIAAIC row was merged in.
        others = [r["title"] for r in recs[1:4] if r["title"] != c["title"]]
        c["also_reported_as"] = " | ".join(others)
        c["emb_row"] = int(sub.index[0] if recs[0]["uid"] == sub.iloc[0]["uid"]
                           else sub.index[sub["uid"] == recs[0]["uid"]][0])
        canon_rows.append(c)
    canon = pd.DataFrame(canon_rows).reset_index(drop=True)
    Ec = E[canon["emb_row"].to_numpy()]

    # Topic clusters over canonical incidents: k-means on unit vectors (spherical k-means).
    # HDBSCAN left 60-80% of incidents as noise on this data at every setting tried, and
    # silhouette was flat (~0.06) for k in 20..80, so k is chosen for readability.
    cl, names = cluster_topics(Ec, canon.title.tolist())
    canon["cluster"] = cl
    canon["cluster_name"] = names
    # 2-D layout for the dashboard scatter.
    xy = PCA(n_components=2, random_state=0).fit_transform(Ec)
    canon["x"], canon["y"] = xy[:, 0].round(4), xy[:, 1].round(4)
    np.save(PROC / "canonical_emb.npy", Ec)
    canon.drop(columns=["emb_row"]).to_parquet(PROC / "canonical.parquet", index=False)

    stats = {
        "records": int(len(df)), "canonical": int(len(canon)),
        "records_by_source": df["source"].value_counts().to_dict(),
        "canonical_by_primary_source": canon["source"].value_counts().to_dict(),
        "aiaaic_merged_into_aiid": int(sum(1 for c in canon_rows
                                           if c["source"] == "aiid" and "aiaaic" in c["sources"])),
        "news_linked_to_known_incident": int(sum(c["community_mentions"] for c in canon_rows
                                                 if c["source"] in ("aiid", "aiaaic"))),
        "n_clusters": int(len(set(cl)) - (1 if -1 in cl else 0)),
        "unclustered_frac": round(float((cl == -1).mean()), 3),
        "candidate_pairs": int(len(decided)),
        "candidate_pairs_judged_same": int(decided["same"].sum()) if len(decided) else 0,
        "judge_coverage": round(float(decided["judged"].mean()), 3) if len(decided) else None,
        "judge_model": JUDGE_MODEL,
        "thresholds": {"T_CAND_INC": T_CAND_INC, "T_CAND_NEWS": T_CAND_NEWS, "T_FALLBACK": T_FALLBACK, "TOPK": TOPK},
    }
    write_json(PROC / "dedupe_stats.json", stats, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
