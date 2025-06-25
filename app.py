#!/usr/bin/env python3
import os
import json
import time
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = '/app/data/input'
OUTPUT_FOLDER = '/app/data/output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'bmp', 'tiff'}

ocr_instances = {}
supported_languages = ["en", "es"]
default_lang = "es"
ocr_initialized = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_ocr():
    global ocr_instances, ocr_initialized
    
    if ocr_initialized:
        return True
    
    try:
        print("🚀 Inicializando PaddleOCR (configuración SIMPLE)...")
        from paddleocr import PaddleOCR
        
        ocr_instances["es"] = PaddleOCR(lang='es')
        ocr_instances["en"] = PaddleOCR(lang='en')
        
        ocr_initialized = True
        print("✅ OCR inicializado exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_ocr_instance(language=None):
    global ocr_instances, ocr_initialized
    
    if not ocr_initialized:
        if not initialize_ocr():
            return None
    
    lang = language or default_lang
    return ocr_instances.get(lang, ocr_instances.get("es"))

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'ocr_ready': ocr_initialized})

@app.route('/init')
def init_models():
    try:
        success = initialize_ocr()
        return jsonify({
            'success': success,
            'models_loaded': list(ocr_instances.keys()) if success else []
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_file():
    try:
        if not ocr_initialized:
            if not initialize_ocr():
                return jsonify({'error': 'OCR not initialized'}), 503
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        language = request.form.get('language', default_lang)
        ocr = get_ocr_instance(language)
        
        if ocr is None:
            return jsonify({'error': 'OCR not available'}), 503
        
        filename = secure_filename(file.filename)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
            file.save(tmp_file.name)
            
            if filename.lower().endswith('.pdf'):
                from pdf2image import convert_from_path
                pages = convert_from_path(tmp_file.name, dpi=300, first_page=1, last_page=1)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as img_tmp:
                    pages[0].save(img_tmp.name, 'JPEG', quality=95)
                    # VOLVER AL MÉTODO QUE FUNCIONABA: .predict()
                    result = ocr.predict(img_tmp.name)
                    os.remove(img_tmp.name)
            else:
                # VOLVER AL MÉTODO QUE FUNCIONABA: .predict()
                result = ocr.predict(tmp_file.name)
            
            os.remove(tmp_file.name)
        
        # VOLVER AL MÉTODO ORIGINAL que funcionaba, pero con pequeñas mejoras
        text_lines = []
        if result and isinstance(result, list) and len(result) > 0:
            page_result = result[0]
            if 'rec_texts' in page_result:
                text_lines = page_result['rec_texts']
        
        # AGREGAMOS: también intentar extraer confianzas si están disponibles
        confidences = []
        if result and isinstance(result, list) and len(result) > 0:
            page_result = result[0]
            if 'rec_scores' in page_result:
                confidences = page_result['rec_scores']
        
        # Calcular confianza promedio
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return jsonify({
            'success': True,
            'text': '\n'.join(text_lines),
            'total_blocks': len(text_lines),
            'filename': filename,
            'language': language,
            'avg_confidence': round(avg_confidence, 3) if avg_confidence > 0 else None,
            'confidence_available': len(confidences) > 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print("🚀 PaddleOCR Simple Server iniciando...")
    print("🔄 Volviendo al método .predict() que funcionaba")
    app.run(host='0.0.0.0', port=8501, debug=False)
