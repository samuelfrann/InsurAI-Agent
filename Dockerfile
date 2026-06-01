FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the HuggingFace embeddings model during build (not at runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy app code
COPY app.py .
COPY ingest.py .
COPY create_user.py .
COPY backend/ backend/

# Copy pre-built data (these are NOT in GitHub — copy from local)
COPY models/ models/
COPY chroma_db/ chroma_db/

# Create directories that the app expects
RUN mkdir -p insurai_memory

# HF Spaces uses port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]