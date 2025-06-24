FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PADDLE_HOME=/opt/paddle_models
ENV OMP_NUM_THREADS=1
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc g++ make cmake \
    libopenblas-dev liblapack-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 \
    poppler-utils curl wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias en orden específico para PaddleOCR 3.0
RUN pip install --no-cache-dir --upgrade pip

# Primero las dependencias base
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    opencv-python-headless==4.8.1.78 \
    Pillow==10.0.1

# PaddlePaddle 3.0 estable
RUN pip install --no-cache-dir \
    paddlepaddle==3.0.0b1

# PaddleOCR 3.0 compatible  
RUN pip install --no-cache-dir \
    paddleocr==3.0.1

# Otras dependencias
RUN pip install --no-cache-dir \
    pdf2image==1.17.0 \
    flask==3.0.0 \
    gunicorn==21.2.0

# Copiar código
COPY . /app/

# Crear directorios
RUN mkdir -p /app/data/input /app/data/output /opt/paddle_models /root/.paddlex

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s \
    CMD curl -f http://localhost:8501/health || exit 1

CMD ["python", "app.py"]
