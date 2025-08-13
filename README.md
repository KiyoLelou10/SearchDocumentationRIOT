# SearchDocumentationRIOT

Hybrid semantic + lexical search for **local RIOT OS Doxygen docs**.
The indexer splits HTML into anchored chunks and builds **FAISS** (semantic) and **BM25** (lexical) indices. A Flask server embeds your query, fuses FAISS+BM25 via **RRF**, reranks with **MiniLM**, and groups by page. The UI shows one link per HTML file; clicking opens the local page at the best anchor.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)

   * [1) Generate RIOT Docs](#1-generate-riot-docs)
   * [2) Create the Index](#2-create-the-index)
   * [3) Search from CLI](#3-search-from-cli)
   * [4) Use the Web UI](#4-use-the-web-ui)
3. [How It Works](#how-it-works)
4. [Index Folder Layout](#index-folder-layout)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

* **Python 3.8+**
* **Doxygen** to generate RIOT’s HTML docs
* Required Python packages:

  ```bash
  pip install numpy faiss-cpu sentence-transformers rank-bm25 \
              beautifulsoup4 lxml tqdm flask
  ```

  > Tip: If you have GPU FAISS, you can use `faiss-gpu` instead of `faiss-cpu`.

---

## Quick Start

### 1) Generate RIOT Docs

Clone RIOT and build Doxygen HTML (paths may vary):

```bash
git clone https://github.com/RIOT-OS/RIOT.git
cd RIOT
doxygen Doxyfile
# HTML typically in: ./doc/doxygen/html
```

### 2) Create the Index

Produce embeddings + indices into `./riot_index`:

```bash
python3 RIOTSearchEmbed.py \
  --docs_dir "path/to/RIOT/doc/doxygen/html" \
  --out_dir ./riot_index \
  --embed_model intfloat/e5-base-v2 \
  --chunk_tokens 60 \
  --chunk_overlap 15
```

### 3) Search from CLI

Run a hybrid search with reranking (pretty output):

```bash
python3 RIOTSearch.py \
  --index_dir ./riot_index \
  --query "Build a DHCPv6 client that requests IA_NA and IA_PD" \
  --pretty \
  --rerank
```

### 4) Use the Web UI

Launch the Flask server and visit the UI:

```bash
python3 RIOTSearchServer.py
# then open http://127.0.0.1:5000
```

* Enter a query; results show **one entry per HTML file** (filename + snippet).
* Clicking a result opens the **local HTML** (file://…) at the best anchor.

---

## How It Works

1. **Chunking**: Each HTML page is split into anchored chunks (title, anchor, text, path).
2. **Semantic index (FAISS)**: SentenceTransformer embeddings enable conceptual matches.
3. **Lexical index (BM25)**: Keyword scoring captures exact-term relevance.
4. **Fusion (RRF)**: Combines FAISS + BM25 candidate lists into one ranking.
5. **Rerank (MiniLM)**: Cross-encoder refines the top results for precision.
6. **Diversification**: Results are grouped by page; the best anchor per page is shown.

---

## Index Folder Layout

After embedding, `./riot_index` contains:

```
manifest.json      # embed model + settings used at index time
metadata.jsonl     # one JSON line per chunk (source_path, title, anchor, text, …)
index.faiss        # dense vector (semantic) index
bm25.pkl           # lexical BM25 index
```

---

## Troubleshooting

* **Links show “Data Fields / Detailed Description”**
  That’s the chunk header text. The UI is configured to display the **filename** as the link label; ensure your template uses `{{ r.display_title }}` or `{{ r.basename }}`.

* **Nothing opens on click**
  The server opens local files with `webbrowser.open_new_tab()`. Make sure `source_path` in `metadata.jsonl` points to existing HTML files (absolute paths are safest on Windows).

* **Slow first query**
  The first request loads models (SentenceTransformer + cross-encoder). Subsequent queries are faster.
