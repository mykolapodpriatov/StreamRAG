# StreamRAG

StreamRAG is a RAG system for streaming news and financial data. It uses Celery for asynchronous ingestion from RSS feeds and other streaming sources, Qdrant for vector storage (with time-decayed scoring), and LangChain for the RAG pipeline.

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
