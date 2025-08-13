#!/usr/bin/env python3
# search_riot_docs.py
"""
Hybrid semantic search over the RIOT OS docs index produced by index_riot_docs.py.

Pipeline
--------
1) Embed the user query (E5-style if applicable).
2) Retrieve top-N by FAISS (cosine/IP).
3) Retrieve top-M by BM25.
4) Fuse results via Reciprocal Rank Fusion (RRF).
5) Optional reranking via a cross-encoder (MiniLM) for extra precision.
6) Dynamic thresholding: keep items above --min_score and before the "elbow" in score drop.
7) Group results by page (source_path) and return at most K chunks per page.

Usage
-----
  python search_riot_docs.py --index_dir ./riot_index --query "How to build a DHCPv6 client?"

Options:
  --max_candidates: number of FAISS/BM25 candidates before fusion
  --max_results: cap on final results after filtering (default 20)
  --min_score: absolute cosine sim cutoff (normalized vectors), e.g., 0.28–0.35 often reasonable for MiniLM/E5
  --rerank: enable cross-encoder reranking (slower, but better precision)
  --pretty: print human-friendly output
  --as_json: emit machine-readable JSON
"""

import argparse
import os
import re
import sys
import json
import pickle
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

def load_manifest(index_dir: str) -> Dict[str, Any]:
    with open(os.path.join(index_dir, "manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def build_query_embedding(model_name: str, query: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    if "e5" in model_name.lower() or "bge" in model_name.lower():
        enc = model.encode(["query: " + query], normalize_embeddings=True, convert_to_numpy=True)
    else:
        enc = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return enc.astype("float32")

def faiss_search(index_path: str, qvec: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    index = faiss.read_index(index_path)
    D, I = index.search(qvec, topk)
    return D[0], I[0]

def rrf_fusion(runs: List[List[int]], k: int = 60) -> List[int]:
    """
    Reciprocal Rank Fusion: higher is better.
    run[i] is a list of doc ids ordered by that run (0-based rank).
    """
    scores = defaultdict(float)
    for run in runs:
        for rank, doc_id in enumerate(run):
            scores[doc_id] += 1.0 / (k + rank + 1)
    # return doc_ids ranked by fused score
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

def elbow_cutoff(scores: List[float], min_score: float, drop_ratio: float = 0.30) -> int:
    """
    Given a list of sorted similarity scores (descending), return the cutoff index:
    - Stop when score < min_score
    - Or when relative drop between consecutive scores exceeds drop_ratio
    Returns the number of items to keep.
    """
    keep = 0
    prev = None
    for i, s in enumerate(scores):
        if s < min_score:
            break
        if prev is not None and prev > 0:
            rel_drop = (prev - s) / prev
            if rel_drop > drop_ratio and i > 3:  # ignore very top where drops are natural
                break
        keep += 1
        prev = s
    return keep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True, help="Folder containing manifest.json, index.faiss, metadata.jsonl, bm25.pkl")
    ap.add_argument("--query", required=True, help="User query")
    ap.add_argument("--max_candidates", type=int, default=60, help="How many FAISS/BM25 candidates prior to fusion")
    ap.add_argument("--max_results", type=int, default=20, help="Cap on how many results to return after filtering")
    ap.add_argument("--min_score", type=float, default=0.30, help="Absolute cosine similarity cutoff")
    ap.add_argument("--rerank", action="store_true", help="Use cross-encoder reranking (MiniLM)")
    ap.add_argument("--pretty", action="store_true", help="Print human-friendly output")
    ap.add_argument("--as_json", action="store_true", help="Emit JSON")
    ap.add_argument("--max_per_page", type=int, default=2, help="At most N chunks per source page")
    args = ap.parse_args()

    manifest = load_manifest(args.index_dir)
    meta_path = os.path.join(args.index_dir, "metadata.jsonl")
    faiss_path = os.path.join(args.index_dir, "index.faiss")
    bm25_path = os.path.join(args.index_dir, "bm25.pkl")

    meta = read_jsonl(meta_path)
    with open(bm25_path, "rb") as f:
        bm25: BM25Okapi = pickle.load(f)

    embed_model = manifest["embed_model"]
    qvec = build_query_embedding(embed_model, args.query)

    # FAISS candidates
    D, I = faiss_search(faiss_path, qvec, args.max_candidates)
    faiss_order = [i for i in I if i >= 0]

    # BM25 candidates
    query_tokens = re.findall(r"\w+", args.query.lower())
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_order = list(np.argsort(-bm25_scores))[:args.max_candidates]

    # Fuse with RRF
    fused = rrf_fusion([faiss_order, bm25_order])

    # Map fused -> similarities (from FAISS) for cutoff
    sim_map = {int(idx): float(D[list(I).index(idx)]) if idx in set(I) else 0.0 for idx in fused if idx >= 0}
    # Sorted by similarity
    fused_sorted = sorted(fused, key=lambda x: sim_map.get(int(x), 0.0), reverse=True)
    sims_sorted = [sim_map.get(int(x), 0.0) for x in fused_sorted]

    keep_n = elbow_cutoff(sims_sorted, min_score=args.min_score, drop_ratio=0.30)
    trimmed = fused_sorted[:max(keep_n, 1)]
    trimmed = trimmed[:args.max_results * 3]  # leave room for rerank + grouping

    # Optional reranking with a cross-encoder
    if args.rerank:
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(args.query, meta[i]["text"]) for i in trimmed]
        ce_scores = ce.predict(pairs).tolist()
        trimmed = [x for _, x in sorted(zip(ce_scores, trimmed), key=lambda t: t[0], reverse=True)]

    # Group by page and cap per page
    grouped = defaultdict(list)
    for idx in trimmed:
        rec = meta[int(idx)]
        grouped[rec["source_path"]].append((idx, sim_map.get(int(idx), 0.0)))

    # For each page, sort by similarity and keep top N
    final = []
    for page, items in grouped.items():
        items = sorted(items, key=lambda x: x[1], reverse=True)[:args.max_per_page]
        final.extend(items)

    # Global sort & cap
    final = sorted(final, key=lambda x: x[1], reverse=True)[:args.max_results]

    results = []
    for idx, score in final:
        rec = meta[int(idx)]
        results.append({
            "score": float(score),
            "title": rec["title"],
            "anchor": rec["anchor"],
            "source_path": rec["source_path"],
            "snippet": rec["text"][:700] + ("..." if len(rec["text"]) > 700 else ""),
            "chunk_hash": rec.get("chunk_hash", ""),
        })

    if args.as_json:
        print(json.dumps({"query": args.query, "results": results}, ensure_ascii=False, indent=2))
        return

    if args.pretty:
        print(f"\nQuery: {args.query}\n")
        for i, r in enumerate(results, 1):
            anchor = ("#" + r["anchor"]) if r["anchor"] else ""
            print(f"[{i}] score={r['score']:.3f}  {r['title']}")
            print(f"     {r['source_path']}{anchor}")
            print("     " + re.sub(r"\s+", " ", r["snippet"])[:180])
            print()
    else:
        # Minimal
        for r in results:
            print(f"{r['score']:.3f}\t{r['title']}\t{r['source_path']}#{r['anchor']}")

if __name__ == "__main__":
    main()
