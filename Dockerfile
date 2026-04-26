FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_MODE=api \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    STREAMLIT_PORT=8501

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY src ./src
COPY models ./models
COPY streamlit_app.py ./streamlit_app.py

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# Default mode starts FastAPI.
# To run Streamlit instead:
#   docker run -e APP_MODE=streamlit -p 8501:8501 telco-churn-api
CMD ["sh", "-c", "if [ \"$APP_MODE\" = \"streamlit\" ]; then streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port ${STREAMLIT_PORT} --server.headless true; else uvicorn src.app:app --host ${API_HOST} --port ${API_PORT}; fi"]
