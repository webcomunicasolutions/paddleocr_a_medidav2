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

# Instalar Python packages con versiones específicas compatibles
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    opencv-python-headless==4.8.1.78 \
    paddlepaddle==2.6.1 \
    paddleocr==2.7.3 \
    pdf2image==1.17.0 \
    flask==3.0.0 \
    Pillow==10.0.1

# Copiar código
COPY . /app/

# Crear directorios
RUN mkdir -p /app/data/input /app/data/output /opt/paddle_models

# NO pre-descargamos modelos aquí para evitar conflictos
# Los modelos se descargarán en el primer uso

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -f http://localhost:8501/health || exit 1

CMD ["python", "app.py"]
