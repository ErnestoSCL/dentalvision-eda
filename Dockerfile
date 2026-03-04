# ── Base image ────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Dependencias del sistema ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencias Python ─────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Código de la app ────────────────────────────────────────────────────
COPY . .

# ── Puerto requerido por Hugging Face Spaces ────────────────────────────
EXPOSE 7860

# ── Comando de inicio ────────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
