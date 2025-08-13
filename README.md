# SearchDocumentationRIOT
Hybrid search for local RIOT OS docs. The indexer splits Doxygen HTML into anchored chunks, stores metadata and builds FAISS (semantic) and BM25 (lexical) indexes. A Flask server embeds your query, fuses FAISS+BM25 via RRF, reranks with MiniLM, groups results by page. The UI shows filename and snippet; clicking opens local HTML at the best anchor.!

# Prerequisites
Clone into RIOT OS and generate the doxygen
Also install all necessary dependencies

# Embedder
To create the embeddings run python3 RIOTSearchEmbed.py --docs_dir "your\path\to\RIOT\doc\doxygen\html"  --out_dir ./riot_index --embed_model intfloat/e5-base-v2    --chunk_tokens 60 --chunk_overlap 15

# Search
If you want to search without using the UI, you can just run python3 RIOTSearch.py --index_dir ./riot_index --query "Build a DHCPv6 client that requests IA_NA and IA_PD" --pretty --rerank

# UI
You can also just use the UI which is a simple html, which displays search results you can click on, so the html page from your documentation folder is opened run python3 RIOTSearchServer.py and then visit http://127.0.0.1:5000

 
