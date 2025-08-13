#!/usr/bin/env python3
# index_riot_docs.py
"""
Build a high-quality local retrieval index for RIOT OS docs (HTML, e.g., Doxygen output).

What it does
------------
- Walk a docs directory (your local HTML folder).
- Parse each HTML page, split by headings (h1–h4), and then chunk long sections with overlap.
- Compute dense embeddings (default: E5-base-v2) for each chunk.
- Build a FAISS index (cosine similarity via L2-normalized vectors).
- Build a BM25 index (rank_bm25) over the same chunks for robust hybrid retrieval.
- Persist everything on disk so you can query fast.

Usage
-----
  python index_riot_docs.py --docs_dir "C:/path/to/RIOT/doc/doxygen/html" --out_dir ./riot_index

You can swap the embedding model with --embed_model. Good options:
 - intfloat/e5-base-v2 (recommended; remember to use "query: ..." / "passage: ...")
 - thenlper/gte-base
 - BAAI/bge-base-en-v1.5  (prefix with "query: " / "passage: ")
"""

import argparse
import os
import re
import sys
import json
import pickle
import hashlib
from typing import List, Dict, Any, Tuple

from bs4 import BeautifulSoup
from tqdm import tqdm

# Embeddings + FAISS + BM25
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ------------------------------
# Text extraction & chunking
# ------------------------------

HEADING_TAGS = ["h1", "h2", "h3", "h4"]

def clean_text(s: str) -> str:
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def extract_sections_from_html(html: str, source_path: str) -> List[Dict[str, Any]]:
    """
    Returns a list of sections in reading order.
    Each section has: {title, anchor, level, text, source_path}
    We remove nav/script/style and keep code/pre blocks as text.
    """
    soup = BeautifulSoup(html, "lxml")
    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    body = soup.body or soup
    sections = []
    current = None

    def start_section(h):
        title = clean_text(h.get_text(" ", strip=True))
        anchor = h.get("id") or h.get("name") or ""
        level = int(h.name[1]) if h.name and h.name[0].lower() == 'h' and h.name[1].isdigit() else 1
        return {"title": title, "anchor": anchor, "level": level, "text_parts": [], "source_path": source_path}

    for elem in body.descendants:
        if not getattr(elem, "name", None):
            continue
        name = elem.name.lower()
        if name in HEADING_TAGS:
            if current:
                text = clean_text(" ".join(current["text_parts"]))
                if text:
                    current["text"] = text
                    sections.append({k: v for k, v in current.items() if k != "text_parts"})
            current = start_section(elem)
        elif name in ("p", "li"):
            if current:
                current["text_parts"].append(clean_text(elem.get_text(" ", strip=True)))
        elif name in ("pre", "code"):
            if current:
                # Keep code blocks – they matter in API docs
                current["text_parts"].append("\n```\n" + elem.get_text("\n", strip=True) + "\n```\n")

    # Flush last
    if current:
        text = clean_text(" ".join(current["text_parts"]))
        if text:
            current["text"] = text
            sections.append({k: v for k, v in current.items() if k != "text_parts"})

    # If no headings found, fallback to full body text
    if not sections:
        text = clean_text(body.get_text(" ", strip=True))
        if text:
            sections = [{
                "title": os.path.basename(source_path),
                "anchor": "",
                "level": 1,
                "text": text,
                "source_path": source_path,
            }]

    return sections

def approx_tokenize(s: str) -> List[str]:
    # Lightweight sentence-ish splitter to keep structure; avoids heavy deps.
    # We'll later pack these into ~600-token chunks with ~100-token overlap.
    s = s.replace("\r", " ")
    parts = re.split(r'(?<=[\.\?\!])\s+|\n+', s)
    tokens = []
    for p in parts:
        p = p.strip()
        if p:
            tokens.append(p)
    return tokens

def chunk_text(text: str, target_tokens: int = 600, overlap: int = 100) -> List[str]:
    sents = approx_tokenize(text)
    if not sents:
        return []
    chunks = []
    i = 0
    while i < len(sents):
        cur = sents[i:i+target_tokens]
        chunks.append(" ".join(cur))
        if i + target_tokens >= len(sents):
            break
        i += max(1, target_tokens - overlap)
    return chunks

# ------------------------------
# Embedding & indexing
# ------------------------------

def build_embeddings(model_name: str, texts: List[str], is_passage: bool = True, batch_size: int = 32) -> np.ndarray:
    model = SentenceTransformer(model_name)
    to_encode = []
    if "e5" in model_name.lower() or "bge" in model_name.lower():
        prefix = "passage: " if is_passage else "query: "
        to_encode = [prefix + t for t in texts]
    else:
        to_encode = texts
    embs = model.encode(to_encode, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    return embs.astype("float32")

def save_faiss_index(vectors: np.ndarray, path: str):
    # Cosine sim with FAISS: store normalized vectors and use IndexFlatIP
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, path)

def load_html(path: str) -> str:
    with open(path, "rb") as f:
        return f.read().decode(errors="ignore")

def hash_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_dir", required=True, help="Folder with RIOT HTML docs (e.g., .../doc/doxygen/html)")
    ap.add_argument("--out_dir", default="./riot_index", help="Where to write the index")
    ap.add_argument("--embed_model", default="intfloat/e5-base-v2", help="Sentence-transformers model name")
    ap.add_argument("--chunk_tokens", type=int, default=600, help="Approx tokens per chunk")
    ap.add_argument("--chunk_overlap", type=int, default=100, help="Overlap between chunks")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, "metadata.jsonl")
    vec_path  = os.path.join(args.out_dir, "index.faiss")
    bm25_path = os.path.join(args.out_dir, "bm25.pkl")

    # Collect HTML files
    html_files = []
    for root, _, files in os.walk(args.docs_dir):
        for fn in files:
            if fn.lower().endswith(".html"):
                html_files.append(os.path.join(root, fn))
    html_files.sort()

    print(f"Found {len(html_files)} HTML files. Parsing & chunking...")

    all_chunks = []
    all_chunk_meta = []  # aligned with all_chunks
    for fp in tqdm(html_files):
        try:
            html = load_html(fp)
        except Exception as e:
            print(f"[warn] failed to read {fp}: {e}")
            continue
        sections = extract_sections_from_html(html, fp)
        for sec in sections:
            sec_text = sec.get("text", "").strip()
            if not sec_text:
                continue
            chunks = chunk_text(sec_text, target_tokens=args.chunk_tokens, overlap=args.chunk_overlap)
            for ch in chunks:
                if len(ch) < 40:  # skip tiny chunks
                    continue
                all_chunks.append(ch)
                all_chunk_meta.append({
                    "source_path": sec["source_path"],
                    "title": sec["title"],
                    "anchor": sec["anchor"],
                    "level": sec["level"],
                    "chunk_hash": hash_text(ch),
                })

    if not all_chunks:
        print("No chunks produced. Check your docs directory.")
        sys.exit(1)

    print(f"Prepared {len(all_chunks)} chunks. Building BM25 corpus...")
    # Tokenize for BM25 (simple whitespace/token split)
    tokenized_corpus = [re.findall(r"\w+", t.lower()) for t in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    print("Computing embeddings...")
    vectors = build_embeddings(args.embed_model, all_chunks, is_passage=True, batch_size=64)
    print("Saving FAISS index...")
    save_faiss_index(vectors, vec_path)

    print("Saving metadata & BM25...")
    with open(meta_path, "w", encoding="utf-8") as f:
        for meta, text in zip(all_chunk_meta, all_chunks):
            rec = dict(meta)
            rec["text"] = text
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    # Store a small manifest
    manifest = {
        "embed_model": args.embed_model,
        "num_chunks": len(all_chunks),
        "faiss_index": vec_path,
        "metadata": meta_path,
        "bm25": bm25_path,
        "chunk_tokens": args.chunk_tokens,
        "chunk_overlap": args.chunk_overlap,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Done. Index written to", args.out_dir)


if __name__ == "__main__":
    main()
