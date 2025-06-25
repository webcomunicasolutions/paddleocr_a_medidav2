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
        
        # MANTENER configuración simple que ya funciona
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

def extract_text_improved(ocr_result):
    """
    PASO 1: Mejorar la extracción de texto sin cambiar configuración OCR
    Solo mejoramos cómo procesamos los resultados que ya funcionan
    """
    if not ocr_result or not isinstance(ocr_result, list):
        return {
            'text': '',
            'blocks': [],
            'total_blocks': 0,
            'confidences': []
        }
    
    all_text_lines = []
    text_blocks = []
    confidences = []
    
    try:
        # Procesar el resultado del OCR
        for page_result in ocr_result:
            if not page_result:
                continue
                
            for block in page_result:
                try:
                    # Estructura típica: [coordenadas, (texto, confianza)]
                    if len(block) >= 2:
                        coordinates = block[0] if len(block) > 0 else []
                        text_info = block[1] if len(block) > 1 else ('', 0.0)
                        
                        # Extraer texto y confianza
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = str(text_info[0]).strip()
                            confidence = float(text_info[1]) if text_info[1] else 0.0
                        else:
                            text = str(text_info).strip()
                            confidence = 1.0
                        
                        if text:  # Solo si hay texto
                            all_text_lines.append(text)
                            confidences.append(confidence)
                            
                            # Crear bloque básico con info adicional
                            text_blocks.append({
                                'text': text,
                                'confidence': round(confidence, 3),
                                'coordinates': coordinates
                            })
                            
                except Exception as e:
                    print(f"⚠️ Error procesando bloque: {e}")
                    continue
                    
    except Exception as e:
        print(f"⚠️ Error general extrayendo texto: {e}")
    
    # Texto combinado
    combined_text = '\n'.join(all_text_lines)
    
    # Estadísticas básicas
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    return {
        'text': combined_text,
        'blocks': text_blocks,
        'total_blocks': len(text_blocks),
        'confidences': confidences,
        'avg_confidence': round(avg_confidence, 3),
        'min_confidence': round(min(confidences), 3) if confidences else 0.0,
        'max_confidence': round(max(confidences), 3) if confidences else 0.0
    }

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
        detailed = request.form.get('detailed', 'false').lower() == 'true'
        
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
                    # CAMBIO: usar .ocr() en lugar de .predict() para más información
                    result = ocr.ocr(img_tmp.name)
                    os.remove(img_tmp.name)
            else:
                # CAMBIO: usar .ocr() en lugar de .predict()
                result = ocr.ocr(tmp_file.name)
            
            os.remove(tmp_file.name)
        
        # CAMBIO: usar nuestra función mejorada de extracción
        processed_result = extract_text_improved(result)
        
        # Respuesta básica
        response = {
            'success': True,
            'text': processed_result['text'],
            'total_blocks': processed_result['total_blocks'],
            'avg_confidence': processed_result['avg_confidence'],
            'filename': filename,
            'language': language
        }
        
        # Si piden detalle, agregar más info
        if detailed:
            response.update({
                'blocks': processed_result['blocks'],
                'min_confidence': processed_result['min_confidence'],
                'max_confidence': processed_result['max_confidence'],
                'confidences': processed_result['confidences']
            })
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print("🚀 PaddleOCR Simple Server iniciando...")
    print("📈 PASO 1: Extracción mejorada de texto y confianza")
    app.run(host='0.0.0.0', port=8501, debug=False)
