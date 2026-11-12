# Streamlit frontend. No model and no torch here: it only calls the API.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src \
    UV_SYSTEM_PYTHON=1 UV_NO_CACHE=1 STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml ./
RUN uv pip install -r pyproject.toml --extra frontend

COPY src ./src
RUN useradd --create-home app
USER app

EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "src/jass_cv/frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
