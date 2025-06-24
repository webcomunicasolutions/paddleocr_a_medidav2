#!/usr/bin/env python3
import os
import json
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from paddleocr import PaddleOCR
from pdf2image import convert_from_path

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = '/app/data/input'
OUTPUT_FOLDER = '/app/data/output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'bmp', 'tiff'}

# OCR instances
ocr_en = PaddleOCR(lang='en', use_gpu=False, show_log=False)
ocr_es = PaddleOCR(lang='es', use_gpu=False, show_log=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_ocr(language='es'):
    return ocr_es if language == 'es' else ocr_en

@app.route('/health')
def health():
    """Health check"""
    try:
        # Test básico
        import numpy as np
        test_img = np.ones((50, 100, 3), dtype=np.uint8) * 255
        ocr_es.ocr(test_img, cls=False)
        return jsonify({'status': 'healthy', 'languages': ['en', 'es']})
    except:
        return jsonify({'status': 'unhealthy'}), 503

@app.route('/')
def index():
    return jsonify({
        'service': 'PaddleOCR Server',
        'endpoints': ['/health', '/process', '/status'],
        'languages': ['en', 'es']
    })

@app.route('/process', methods=['POST'])
def process_file():
    """Procesar archivo"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        language = request.form.get('language', 'es')
        
        # Guardar archivo
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Convertir PDF si es necesario
        if filename.lower().endswith('.pdf'):
            pages = convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
            jpg_path = filepath.replace('.pdf', '.jpg')
            pages[0].save(jpg_path, 'JPEG')
            filepath = jpg_path
        
        # Procesar con OCR
        ocr = get_ocr(language)
        result = ocr.ocr(filepath, cls=True)
        
        # Extraer texto
        text_lines = []
        if result and result[0]:
            for line in result[0]:
                if len(line) >= 2:
                    text_lines.append(line[1][0])
        
        # Respuesta
        response = {
            'success': True,
            'text': '\n'.join(text_lines),
            'raw_result': result,
            'language': language,
            'filename': filename
        }
        
        # Limpiar archivos
        for f in [filepath, filepath.replace('.jpg', '.pdf')]:
            if os.path.exists(f):
                os.remove(f)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    return jsonify({
        'service': 'PaddleOCR Server',
        'version': '1.0',
        'languages': ['en', 'es'],
        'formats': list(ALLOWED_EXTENSIONS)
    })

if __name__ == '__main__':
    # Crear directorios
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("🚀 PaddleOCR Server iniciando...")
    app.run(host='0.0.0.0', port=8501, debug=False)