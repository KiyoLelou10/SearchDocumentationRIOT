# SearchDocumentationRIOT
Hybrid search for local RIOT OS docs. The indexer splits Doxygen HTML into anchored chunks, stores metadata and builds FAISS (semantic) and BM25 (lexical) indexes. A Flask server embeds your query, fuses FAISS+BM25 via RRF, reranks with MiniLM, groups results by page. The UI shows filename and snippet; clicking opens local HTML at the best anchor.!
