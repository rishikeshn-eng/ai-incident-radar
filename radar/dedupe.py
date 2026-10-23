"""Stage 2: embed every record, merge cross-source duplicates, and cluster canonical incidents.

Embeddings: BAAI/bge-base-en-v1.5 via fastembed (ONNX, CPU, 768-d), cached per record
in data/processed/emb_cache.npz so weekly refreshes only embed new records.

Dedupe rule (union-find):
  * AIAAIC -> AIID      cosine >= T_INC and |year diff| <= 1
  * HN/Reddit -> any    cosine >= T_NEWS (story is coverage of a known incident)
  * HN/Reddit <-> HN    cosine >= T_NEWS and within 14 days (same story posted twice)
Canonical record per group: AIID > AIAAIC > community (highest score).
Thresholds are audited in validate.py with an LLM judge on stratified pairs.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import PROC, ROOT, write_json

EMB_MODEL = "BAAI/bge-base-en-v1.5"
DIM = 768
T_INC = 0.86
T_NEWS = 0.84
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


def dedupe(df: pd.DataFrame, E: np.ndarray):
    n = len(df)
    src = df["source"].to_numpy()
    year = df["year"].to_numpy(dtype=float)
    dates = pd.to_datetime(df["date"], errors="coerce")
    is_aiid = src == "aiid"
    is_aiaaic = src == "aiaaic"
    is_news = np.isin(src, ["hn", "reddit"])
    uf = UF(n)
    pairs = []  # (i, j, sim, kind) for the audit
    S_inc = E[is_aiaaic] @ E[is_aiid].T
    ia, ib = np.where(is_aiaaic)[0], np.where(is_aiid)[0]
    for r, i in enumerate(ia):
        j_loc = int(np.argmax(S_inc[r]))
        s = float(S_inc[r, j_loc])
        j = ib[j_loc]
        ok_year = np.isnan(year[i]) or np.isnan(year[j]) or abs(year[i] - year[j]) <= 1
        pairs.append((i, j, s, "aiaaic-aiid", bool(ok_year)))
        if s >= T_INC and ok_year:
            uf.union(i, j)
    inc_idx = np.where(~is_news)[0]
    news_idx = np.where(is_news)[0]
    if len(news_idx):
        S_n = E[news_idx] @ E[inc_idx].T
        for r, i in enumerate(news_idx):
            j = inc_idx[int(np.argmax(S_n[r]))]
            s = float(S_n[r].max())
            pairs.append((i, j, s, "news-incident", True))
            if s >= T_NEWS:
                uf.union(i, j)
        S_nn = E[news_idx] @ E[news_idx].T
        np.fill_diagonal(S_nn, -1)
        for r, i in enumerate(news_idx):
            for c in np.where(S_nn[r] >= T_NEWS)[0]:
                j = news_idx[c]
                if j > i and abs((dates[i] - dates[j]).days) <= 14:
                    uf.union(i, j)
                    pairs.append((i, j, float(S_nn[r, c]), "news-news", True))
    groups = np.array([uf.find(i) for i in range(n)])
    return groups, pairs


def score(r) -> tuple:
    pri = {"aiid": 0, "aiaaic": 1, "hn": 2, "reddit": 3}[r["source"]]
    ex = json.loads(r["extra"] or "{}")
    return (pri, -(ex.get("points") or ex.get("score") or 0))


def label_clusters(texts: list[str], labels: np.ndarray, k: int = 4) -> dict[int, str]:
    """c-TF-IDF style: concatenate docs per cluster, take top distinctive terms."""
    ids = sorted(set(labels) - {-1})
    docs = [" ".join(t for t, l in zip(texts, labels) if l == c) for c in ids]
    if not docs:
        return {}
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_df=0.5,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b", sublinear_tf=True)
    X = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for row, c in enumerate(ids):
        top = vocab[np.argsort(-X[row].toarray()[0])][:k * 2]
        picked = []
        for t in top:  # drop unigrams already inside a picked bigram
            if not any(t in p or p in t for p in picked):
                picked.append(t)
            if len(picked) == k:
                break
        out[c] = " / ".join(picked)
    return out


def main():
    df = pd.read_parquet(PROC / "records.parquet").reset_index(drop=True)
    E = embed([emb_text(r) for r in df.to_dict("records")])
    groups, pairs = dedupe(df, E)
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

    # Topic clusters over canonical incidents. PCA first: HDBSCAN struggles in 768-d.
    Z = PCA(n_components=32, random_state=0).fit_transform(Ec)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    cl = HDBSCAN(min_cluster_size=8, min_samples=4, metric="euclidean").fit_predict(Z)
    canon["cluster"] = cl
    names = label_clusters([f"{t} {x[:300]}" for t, x in zip(canon.title, canon.text)], cl)
    canon["cluster_name"] = [names.get(c, "unclustered") for c in cl]
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
        "thresholds": {"T_INC": T_INC, "T_NEWS": T_NEWS},
    }
    write_json(PROC / "dedupe_stats.json", stats, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
