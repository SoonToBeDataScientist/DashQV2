FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG WITH_FINBERT=false
RUN if [ "$WITH_FINBERT" = "true" ]; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir "transformers>=4.40"; \
    fi

COPY . .
RUN mkdir -p data models

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", \
     "--server.address=0.0.0.0", "--server.headless=true"]
