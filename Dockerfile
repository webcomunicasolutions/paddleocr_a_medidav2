FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PADDLE_HOME=/opt/paddle_models

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc g++ libopenblas-dev liblapack-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 \
    poppler-utils curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar Python packages
RUN pip install --no-cache-dir \
    paddlepaddle==2.6.1 \
    paddleocr==2.7.3 \
    pdf2image==1.17.0 \
    flask==3.0.0

# Copiar código
COPY . /app/

# Pre-descargar modelos (CORREGIDO)
RUN python3 -c "from paddleocr import PaddleOCR; print('Descargando modelos...'); PaddleOCR(lang='en', use_gpu=False, show_log=False); PaddleOCR(lang='es', use_gpu=False, show_log=False); print('Modelos listos!')"

# Crear directorios
RUN mkdir -p /app/data/input /app/data/output

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -f http://localhost:8501/health || exit 1

CMD ["python", "app.py"]
