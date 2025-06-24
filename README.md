# PaddleOCR Server

Servidor OCR con soporte para inglés y español.

## Uso en Easypanel

1. Conectar repositorio GitHub
2. Port: 8501
3. Volumen: /app/data
4. Deploy automático

## API

- `GET /health` - Estado del servidor
- `POST /process` - Procesar archivo
- `GET /status` - Información del servicio

## Ejemplo

```bash
curl -X POST http://tu-dominio.com/process \
  -F "file=@documento.pdf" \
  -F "language=es"
