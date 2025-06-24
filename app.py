#!/usr/bin/env python3
import os
import json
import threading
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = '/app/data/input'
OUTPUT_FOLDER = '/app/data/output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'bmp', 'tiff'}

# Variables globales para OCR instances
ocr_en = None
ocr_es = None
ocr_lock = threading.Lock()
models_loading = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_ocr_instance(language='es'):
    """Inicializa instancia OCR de forma lazy"""
    global ocr_en, ocr_es, models_loading
    
    with ocr_lock:
        if models_loading:
            return None
            
        models_loading = True
        
        try:
            print(f"🔄 Inicializando OCR para {language}...")
            
            if language == 'en' and ocr_en is None:
                from paddleocr import PaddleOCR
                ocr_en = PaddleOCR(lang='en', use_gpu=False, show_log=False)
                print("✅ OCR inglés listo")
                
            elif language == 'es' and ocr_es is None:
                from paddleocr import PaddleOCR
                ocr_es = PaddleOCR(lang='es', use_gpu=False, show_log=False)
                print("✅ OCR español listo")
                
        except Exception as e:
            print(f"❌ Error inicializando OCR {language}: {e}")
            return None
        finally:
            models_loading = False
            
    return ocr_en if language == 'en' else ocr_es

def get_ocr(language='es'):
    """Obtiene instancia OCR, inicializándola si es necesario"""
    global ocr_en, ocr_es
    
    if language == 'en':
        if ocr_en is None:
            ocr_en = init_ocr_instance('en')
        return ocr_en
    else:
        if ocr_es is None:
            ocr_es = init_ocr_instance('es')
        return ocr_es

@app.route('/health')
def health():
    """Health check básico"""
    try:
        # Health check sin cargar modelos pesados
        return jsonify({
            'status': 'healthy', 
            'languages': ['en', 'es'],
            'models_loaded': {
                'en': ocr_en is not None,
                'es': ocr_es is not None
            }
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

@app.route('/warmup')
def warmup():
    """Endpoint para pre-cargar modelos"""
    try:
        print("🔥 Precargando modelos OCR...")
        
        # Cargar ambos modelos
        init_ocr_instance('es')
        init_ocr_instance('en')
        
        return jsonify({
            'status': 'models_loaded',
            'en_ready': ocr_en is not None,
            'es_ready': ocr_es is not None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return jsonify({
        'service': 'PaddleOCR Server',
        'version': '1.0',
        'endpoints': ['/health', '/process', '/status', '/warmup'],
        'languages': ['en', 'es'],
        'models_loaded': {
            'en': ocr_en is not None,
            'es': ocr_es is not None
        }
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
        
        # Obtener instancia OCR (se carga automáticamente si no existe)
        print(f"🔍 Obteniendo OCR para {language}...")
        ocr = get_ocr(language)
        
        if ocr is None:
            return jsonify({'error': f'OCR {language} no disponible'}), 503
        
        # Guardar archivo
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Convertir PDF si es necesario
        if filename.lower().endswith('.pdf'):
            from pdf2image import convert_from_path
            pages = convert_from_path(filepath, dpi=300, first_page=1, last_page=1)
            jpg_path = filepath.replace('.pdf', '.jpg')
            pages[0].save(jpg_path, 'JPEG')
            filepath = jpg_path
        
        # Procesar con OCR
        print(f"🔍 Procesando con OCR...")
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
            'filename': filename,
            'text_blocks': len(result[0]) if result and result[0] else 0
        }
        
        # Limpiar archivos
        for f in [filepath, filepath.replace('.jpg', '.pdf')]:
            if os.path.exists(f):
                os.remove(f)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error procesando: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    return jsonify({
        'service': 'PaddleOCR Server',
        'version': '1.0',
        'languages': ['en', 'es'],
        'formats': list(ALLOWED_EXTENSIONS),
        'models_status': {
            'en': 'loaded' if ocr_en else 'not_loaded',
            'es': 'loaded' if ocr_es else 'not_loaded'
        }
    })

if __name__ == '__main__':
    # Crear directorios
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("🚀 PaddleOCR Server iniciando...")
    print("💡 Los modelos se cargarán en el primer uso")
    print("💡 Usa /warmup para pre-cargar modelos")
    
    app.run(host='0.0.0.0', port=8501, debug=False)
