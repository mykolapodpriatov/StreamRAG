# StreamRAG

StreamRAG is a streaming-ingestion pipeline for RAG over news / RSS sources. A Celery worker
asynchronously fetches and embeds feed items into Qdrant; a Streamlit dashboard monitors the
ingestion. The retrieval / answering layer is the next milestone (see Roadmap).

> **Status:** Ingestion and retrieval both work (Celery + Redis + Qdrant). The dashboard's
> search box returns ranked, linked, dated passages from the indexed feeds. Answer
> generation on top of those passages is the next milestone.

## What works today
- **Async ingestion** — Celery worker pulls RSS feeds and writes embeddings to Qdrant.
- **Safe fetching** — feed URLs are validated (private / loopback / metadata hosts are rejected) with bounded network I/O.
- **Retrieval:** semantic search over the indexed feeds, blended with time decay so a
  perfect match from 2019 does not outrank a good one from this morning.
- **Monitoring UI:** Streamlit dashboard for the ingestion side and the search box.
- **Local stack** — Redis + Qdrant via docker-compose.

## Retrieval

`retrieval.retrieve(query, k)` returns ranked `RetrievedPassage` objects: title, link,
published date, summary, and the similarity, recency and blended scores separately, so a
result can be explained rather than just displayed. Like `scoring`, the module imports
neither Celery nor Streamlit, so the ranking is unit-testable against a fake embedder and
an in-memory Qdrant client.

Two things about how it ranks:

- **Recency is recomputed at query time.** The worker stores a `recency_weight` on each
  point, but that value was true when the entry was indexed and decays out of date from
  that moment. Ranking re-derives it from `published` against now.
- **The candidate pool is wider than the answer.** Blending recency in can promote an
  entry that was not in the top-k by similarity alone, so Qdrant is asked for
  `k * CANDIDATE_MULTIPLIER` hits and the blend re-ranks them.

An entry with no usable date gets `UNKNOWN_DATE_WEIGHT` (0.0). It loses the recency share
of its score but is not excluded: an unparseable date is a gap in the feed's metadata, not
evidence that the entry is old.

Settings live in `config.py`, read once and shared by the worker, the retriever and the
dashboard. `RECENCY_WEIGHT` (default `0.3`) is recency's share of the final score; `0`
ranks purely by similarity. `EMBEDDING_MODEL` is the one that matters: the worker records
it on every point, and a query against a collection written by a different model raises
`EmbeddingModelMismatchError` instead of ranking by a similarity that means nothing.

## Roadmap (planned)
- Answer generation on top of the retrieved passages (LangChain)

## Setup
1. Start Redis and Qdrant:
   ```bash
   docker-compose up -d
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Set `OPENAI_API_KEY` for embeddings. `QDRANT_URL` (default
   `http://localhost:6333`) and `QDRANT_COLLECTION` (default
   `streamrag_entries`) can be overridden as needed.
3. Run the Celery worker:
   ```bash
   celery -A worker.app worker --loglevel=info
   ```
4. Run the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```
