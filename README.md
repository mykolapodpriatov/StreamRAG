# StreamRAG

StreamRAG is a streaming-ingestion pipeline for RAG over news / RSS sources. A Celery worker
asynchronously fetches and embeds feed items into Qdrant; a Streamlit dashboard monitors the
ingestion. The retrieval / answering layer is the next milestone (see Roadmap).

> **Status:** Ingestion pipeline works (Celery + Redis + Qdrant). The RAG query pipeline is
> currently a placeholder and is under active development.

## What works today
- **Async ingestion** — Celery worker pulls RSS feeds and writes embeddings to Qdrant.
- **Safe fetching** — feed URLs are validated (private / loopback / metadata hosts are rejected) with bounded network I/O.
- **Monitoring UI** — Streamlit dashboard for the ingestion side.
- **Local stack** — Redis + Qdrant via docker-compose.

## Roadmap (planned)
- Retrieval + answer generation pipeline (LangChain)
- Time-decayed / recency-weighted scoring for streaming relevance

## Setup
1. Start Redis and Qdrant:
   ```bash
   docker-compose up -d
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Celery worker:
   ```bash
   celery -A worker.app worker --loglevel=info
   ```
4. Run the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```
