FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PADDLE_PDX_MODEL_SOURCE=HuggingFace
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Dependencias del sistema específicas para PaddleOCR 3.0.2
RUN apt-get update && apt-get install -y \
    gcc g++ make cmake \
    gfortran \
    libopenblas-dev liblapack-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libgtk2.0-dev \
    poppler-utils curl wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar PaddlePaddle 3.0.0 (CRÍTICO para PaddleOCR 3.0.2)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir paddlepaddle==3.0.0

# Instalar PaddleOCR 3.0.2 ESTABLE
RUN pip install --no-cache-dir paddleocr==3.0.2

# Otras dependencias
RUN pip install --no-cache-dir \
    pdf2image==1.17.0 \
    flask==3.0.0 \
    Pillow==10.0.1

# Copiar código
COPY . /app/

# Crear directorios
RUN mkdir -p /app/data/input /app/data/output /root/.paddleocr

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s \
    CMD curl -f http://localhost:8501/health || exit 1

CMD ["python", "app.py"]
