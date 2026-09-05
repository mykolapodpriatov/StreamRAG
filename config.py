"""Settings shared by the worker, the retriever and the dashboard.

Standard library only, so importing this never drags in Celery, Qdrant,
Streamlit or an embedding client. Every module that needs one of these values
imports it from here.

The embedding model matters most. Querying a collection with a model other than
the one that wrote it returns confident nonsense rather than an error, and the
only way to be sure that cannot happen by accident is for there to be exactly
one place the name is read from.
"""

from __future__ import annotations

import os

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "streamrag_entries")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

#: The embedding model used to write AND to query the collection.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

#: How much of a hit's final score comes from recency rather than similarity,
#: in [0, 1]. This is a news pipeline: a perfect semantic match from 2019
#: usually loses to a good one from this morning. 0 disables the blend and
#: ranks purely by similarity.
RECENCY_WEIGHT = float(os.getenv("RECENCY_WEIGHT", "0.3"))

#: How many candidates to pull from Qdrant per requested result before
#: re-ranking. Recency re-ranking can promote an entry that was not in the
#: top-k by similarity alone, so the candidate pool has to be wider than the
#: answer.
CANDIDATE_MULTIPLIER = int(os.getenv("CANDIDATE_MULTIPLIER", "4"))
