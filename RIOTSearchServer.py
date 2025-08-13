#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RIOT docs hybrid search server (FAISS + BM25 + RRF + MiniLM rerank)

- Hardcoded index dir: ./riot_index
- Always uses cross-encoder reranking
- Returns unique HTML pages (best-scoring anchor per page)
- UI at "/" (server-side render) and JSON API at "/api/search"
- Click a result to open the local HTML file (with #anchor)

Run:
  python riot_search_server.py
Then visit:
  http://127.0.0.1:5000/
"""

import os, re, json, time, pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from flask import Flask, request, render_template, jsonify, redirect, url_for
import webbrowser

# ---------- Configuration ----------
INDEX_DIR = "./riot_index"
MAX_CANDIDATES = 60
MAX_RESULTS = 20
MIN_SCORE = 0.30
DROP_RATIO = 0.30
MAX_PER_PAGE = 2

# ---------- Data loading ----------
def _load_manifest(index_dir: str) -> Dict[str, Any]:
    with open(os.path.join(index_dir, "manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

manifest = _load_manifest(INDEX_DIR)
meta_path = os.path.join(INDEX_DIR, "metadata.jsonl")
bm25_path = os.path.join(INDEX_DIR, "bm25.pkl")
faiss_path = os.path.join(INDEX_DIR, "index.faiss")

meta = _read_jsonl(meta_path)
with open(bm25_path, "rb") as f:
    bm25: BM25Okapi = pickle.load(f)
faiss_index = faiss.read_index(faiss_path)

EMBED_MODEL_NAME = manifest["embed_model"]
embedder = SentenceTransformer(EMBED_MODEL_NAME)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------- Search helpers ----------
def _tokenize_query(q: str) -> List[str]:
    return re.findall(r"\w+", q.lower())

def _embed_query(q: str) -> np.ndarray:
    if "e5" in EMBED_MODEL_NAME.lower() or "bge" in EMBED_MODEL_NAME.lower():
        enc = embedder.encode(["query: " + q], normalize_embeddings=True, convert_to_numpy=True)
    else:
        enc = embedder.encode([q], normalize_embeddings=True, convert_to_numpy=True)
    return enc.astype("float32")

def _rrf_fusion(runs: List[List[int]], k: int = 60) -> List[int]:
    scores = defaultdict(float)
    for run in runs:
        for rank, doc_id in enumerate(run):
            scores[int(doc_id)] += 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

def _elbow_cutoff(scores: List[float], min_score: float, drop_ratio: float) -> int:
    keep = 0
    prev = None
    for i, s in enumerate(scores):
        if s < min_score:
            break
        if prev is not None and prev > 0:
            rel_drop = (prev - s) / prev
            if rel_drop > drop_ratio and i > 3:
                break
        keep += 1
        prev = s
    return keep

def _search_once(query: str) -> List[Dict[str, Any]]:
    qvec = _embed_query(query)
    D, I = faiss_index.search(qvec, MAX_CANDIDATES)
    D, I = D[0], I[0]

    faiss_order = [int(i) for i in I if int(i) >= 0]

    tokens = _tokenize_query(query)
    bm25_scores = bm25.get_scores(tokens)
    bm25_order = list(np.argsort(-bm25_scores))[:MAX_CANDIDATES]

    fused = _rrf_fusion([faiss_order, bm25_order])

    idx_to_rank = {int(i): r for r, i in enumerate(I) if int(i) >= 0}
    sim_map = {int(idx): float(D[idx_to_rank[int(idx)]]) if int(idx) in idx_to_rank else 0.0
               for idx in fused if int(idx) >= 0}

    fused_sorted = sorted(fused, key=lambda x: sim_map.get(int(x), 0.0), reverse=True)
    sims_sorted = [sim_map.get(int(x), 0.0) for x in fused_sorted]

    keep_n = _elbow_cutoff(sims_sorted, min_score=MIN_SCORE, drop_ratio=DROP_RATIO)
    trimmed = fused_sorted[:max(keep_n, 1)]
    trimmed = trimmed[:MAX_RESULTS * 3]

    pairs = [(query, meta[int(i)]["text"]) for i in trimmed]
    ce_scores = cross_encoder.predict(pairs).tolist()
    trimmed = [x for _, x in sorted(zip(ce_scores, trimmed), key=lambda t: t[0], reverse=True)]

    grouped = defaultdict(list)
    for idx in trimmed:
        rec = meta[int(idx)]
        grouped[rec["source_path"]].append((int(idx), sim_map.get(int(idx), 0.0)))

    final = []
    for page, items in grouped.items():
        items = sorted(items, key=lambda x: x[1], reverse=True)[:MAX_PER_PAGE]
        final.extend(items)

    final = sorted(final, key=lambda x: x[1], reverse=True)[:MAX_RESULTS]

    results = []
    for idx, score in final:
        rec = meta[int(idx)]
        results.append({
            "score": float(score),
            "title": rec.get("title", ""),
            "anchor": rec.get("anchor", "") or "",
            "source_path": rec["source_path"],
            "snippet": (rec["text"][:700] + ("..." if len(rec["text"]) > 700 else "")),
            "chunk_hash": rec.get("chunk_hash", ""),
        })
    return results

def _unique_pages(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best = {}
    for r in results:
        p = r["source_path"]
        if p not in best or r["score"] > best[p]["score"]:
            best[p] = r
    ordered, seen = [], set()
    for r in results:
        p = r["source_path"]
        if p in best and p not in seen and best[p] is r:
            ordered.append(r)
            seen.add(p)
    return ordered

# ---------- Flask app ----------
app = Flask(__name__, template_folder="templates2")

@app.route("/", methods=["GET"])
def home():
    q = request.args.get("q", "", type=str)
    results = []
    unique = []
    if q.strip():
        results = _search_once(q.strip())
        unique = _unique_pages(results)
        # Use the HTML filename as the display name
        for r in unique:
            r["basename"] = os.path.basename(r["source_path"])
            r["display_title"] = r["basename"]  # <— this is what the template should render
    return render_template("index.html", query=q, results=unique)

@app.route("/api/search", methods=["GET"])
def api_search():
    q = request.args.get("q", "", type=str).strip()
    if not q:
        return jsonify({"query": q, "pages": []})
    results = _unique_pages(_search_once(q))
    pages = [{
        "title": os.path.basename(r["source_path"]),  # file name as the title
        "name": os.path.basename(r["source_path"]),
        "source_path": r["source_path"],
        "anchor": r["anchor"],
        "score": r["score"]
    } for r in results]
    return jsonify({"query": q, "pages": pages})

@app.route("/open", methods=["GET"])
def open_path():
    path = request.args.get("path", "", type=str)
    anchor = request.args.get("anchor", "", type=str)
    q = request.args.get("q", "", type=str)

    try:
        p = Path(path).resolve()
        if not p.exists():
            return f"[warn] file not found: {p}", 404
        uri = p.as_uri()
        url = uri + (f"#{anchor}" if anchor else "")
        webbrowser.open_new_tab(url)
        time.sleep(0.05)
    except Exception as e:
        return f"[warn] failed to open: {e}", 500

    return redirect(url_for("home", **({"q": q} if q else {})))

if __name__ == "__main__":
    app.run(debug=True)
