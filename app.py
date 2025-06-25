#!/usr/bin/env python3
import os
import json
import time
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = '/app/data/input'
OUTPUT_FOLDER = '/app/data/output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'bmp', 'tiff'}

# Variables globales para OCR
ocr_instances = {}
supported_languages = ["en", "es"]
default_lang = "es"
ocr_initialized = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_ocr():
    """Inicializa PaddleOCR 3.0.2 con sintaxis CORRECTA"""
    global ocr_instances, ocr_initialized
    
    if ocr_initialized:
        return True
    
    try:
        print("🚀 Inicializando PaddleOCR 3.0.2...")
        
        from paddleocr import PaddleOCR
        
        # Configuración CORRECTA para PaddleOCR 3.0.2
        base_config = {
            "device": "cpu",           # CPU específico
            "enable_mkldnn": True,     # Aceleración MKLDNN
            "use_angle_cls": True,     # Clasificación de ángulo
            "det": True,               # Detección
            "rec": True,               # Reconocimiento
            "cls": True                # Clasificación
        }
        
        # Inglés
        print("📚 Cargando modelo inglés...")
        config_en = base_config.copy()
        config_en["lang"] = "en"
        ocr_instances["en"] = PaddleOCR(**config_en)
        print("✅ OCR inglés inicializado")
        
        # Español
        print("📚 Cargando modelo español...")
        config_es = base_config.copy()
        config_es["lang"] = "es"
        try:
            ocr_instances["es"] = PaddleOCR(**config_es)
            print("✅ OCR español inicializado")
        except Exception as e:
            print(f"⚠️ Error con español, usando inglés como fallback: {e}")
            ocr_instances["es"] = ocr_instances["en"]
        
        ocr_initialized = True
        print("🎉 PaddleOCR 3.0.2 completamente inicializado")
        return True
        
    except Exception as e:
        print(f"❌ Error inicializando PaddleOCR: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_ocr_instance(language=None):
    """Obtiene instancia OCR"""
    global ocr_instances, ocr_initialized
    
    if not ocr_initialized:
        if not initialize_ocr():
            return None
    
    lang = language or default_lang
    if lang not in supported_languages:
        lang = default_lang
    
    return ocr_instances.get(lang, ocr_instances.get("en"))

def extract_text_from_result(result):
    """Extrae texto del resultado de PaddleOCR 3.0.2"""
    extracted_data = []
    plain_text_lines = []
    
    try:
        if result and isinstance(result, list) and len(result) > 0:
            # El resultado es una lista con un diccionario
            page_result = result[0]
            
            if 'rec_texts' in page_result and 'rec_scores' in page_result:
                rec_texts = page_result['rec_texts']
                rec_scores = page_result['rec_scores']
                rec_polys = page_result.get('rec_polys', [])
                
                for i, (text, score) in enumerate(zip(rec_texts, rec_scores)):
                    # Coordenadas si están disponibles
                    bbox = []
                    if i < len(rec_polys):
                        bbox = rec_polys[i].tolist() if hasattr(rec_polys[i], 'tolist') else []
                    
                    extracted_data.append({
                        'text': str(text),
                        'confidence': float(score),
                        'bbox': bbox
                    })
                    
                    plain_text_lines.append(str(text))
        
        return {
            'detailed_results': extracted_data,
            'plain_text': '\n'.join(plain_text_lines),
            'total_blocks': len(extracted_data)
        }
        
    except Exception as e:
        print(f"❌ Error extrayendo texto: {e}")
        return {
            'detailed_results': [],
            'plain_text': '',
            'total_blocks': 0
        }

@app.route('/health')
def health():
    """Health check optimizado"""
    return jsonify({
        'status': 'healthy',
        'service': 'PaddleOCR 3.0.2 Server',
        'languages': supported_languages,
        'ocr_ready': ocr_initialized,
        'version': '3.0.2',
        'paddle_version': '3.0.0'
    })

@app.route('/init')
def init_models():
    """Endpoint para inicializar modelos manualmente"""
    try:
        success = initialize_ocr()
        return jsonify({
            'success': success,
            'models_loaded': list(ocr_instances.keys()) if success else [],
            'message': 'PaddleOCR 3.0.2 models initialized successfully' if success else 'Failed to initialize models',
            'version': '3.0.2'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return jsonify({
        'service': 'PaddleOCR 3.0.2 Server',
        'version': '3.0.2',
        'paddle_version': '3.0.0',
        'pdf_support': True,
        'endpoints': {
            'health': '/health',
            'init': '/init (initialize models)',
            'process': '/process (POST)',
            'status': '/status'
        },
        'languages': supported_languages,
        'features': [
            'PP-OCRv5 models',
            'CPU MKLDNN optimization', 
            'PDF processing via image conversion',
            'Multi-language support',
            'HuggingFace model source',
            'Correct 3.0.2 API syntax'
        ]
    })

@app.route('/process', methods=['POST'])
def process_file():
    """Procesar archivo con PaddleOCR 3.0.2 - SINTAXIS CORRECTA"""
    start_time = time.time()
    temp_files = []
    
    try:
        # Inicializar OCR si no está listo
        if not ocr_initialized:
            init_success = initialize_ocr()
            if not init_success:
                return jsonify({'error': 'PaddleOCR 3.0.2 not initialized'}), 503
        
        # Validar archivo
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        language = request.form.get('language', default_lang)
        
        # Obtener OCR
        ocr = get_ocr_instance(language)
        if ocr is None:
            return jsonify({'error': f'OCR {language} not available'}), 503
        
        # Guardar archivo en temporal
        filename = secure_filename(file.filename)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
            file.save(tmp_file.name)
            temp_files.append(tmp_file.name)
            process_path = tmp_file.name
        
        # Procesar según tipo
        if filename.lower().endswith('.pdf'):
            print(f"📄 Procesando PDF: {filename}")
            
            try:
                # Convertir PDF a imagen (método más estable para 3.0.2)
                from pdf2image import convert_from_path
                pages = convert_from_path(process_path, dpi=300, first_page=1, last_page=1)
                
                if not pages:
                    return jsonify({'error': 'No pages in PDF'}), 400
                
                # Guardar como imagen temporal
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as img_tmp:
                    pages[0].save(img_tmp.name, 'JPEG', quality=95)
                    temp_files.append(img_tmp.name)
                    
                    # Procesar con OCR - SINTAXIS CORRECTA 3.0.2
                    print(f"🔍 Ejecutando OCR en imagen convertida...")
                    result = ocr.predict(img_tmp.name)
                    
            except Exception as pdf_error:
                return jsonify({'error': f'PDF processing failed: {str(pdf_error)}'}), 500
        else:
            print(f"🖼️ Procesando imagen: {filename}")
            # Procesar imagen - SINTAXIS CORRECTA 3.0.2
            result = ocr.predict(process_path)
        
        # Extraer y procesar texto
        text_data = extract_text_from_result(result)
        processing_time = time.time() - start_time
        
        # Respuesta estructurada
        response = {
            'success': True,
            'text': text_data['plain_text'],
            'detailed_results': text_data['detailed_results'],
            'language': language,
            'filename': filename,
            'processing_time': round(processing_time, 3),
            'text_blocks_found': text_data['total_blocks'],
            'paddleocr_version': '3.0.2',
            'paddle_version': '3.0.0',
            'api_method': 'predict',
            'timestamp': time.time()
        }
        
        print(f"✅ Procesado en {processing_time:.2f}s - {text_data['total_blocks']} bloques de texto")
        print(f"📄 Texto extraído: {text_data['plain_text'][:100]}...")
        
        return jsonify(response)
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Error procesando: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'processing_time': round(processing_time, 3),
            'paddleocr_version': '3.0.2'
        }), 500
    
    finally:
        # Limpiar archivos temporales
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

@app.route('/status')
def status():
    return jsonify({
        'service': 'PaddleOCR 3.0.2 Server',
        'version': '3.0.2',
        'paddle_version': '3.0.0',
        'languages': supported_languages,
        'formats': list(ALLOWED_EXTENSIONS),
        'models_loaded': list(ocr_instances.keys()),
        'ocr_initialized': ocr_initialized,
        'features': {
            'pp_ocrv5': True,
            'cpu_mkldnn': True,
            'pdf_processing': True,
            'multi_language': True,
            'huggingface_models': True,
            'correct_api_syntax': True
        },
        'config': {
            'model_source': os.environ.get('PADDLE_PDX_MODEL_SOURCE', 'HuggingFace'),
            'api_method': 'predict',
            'mkldnn_enabled': True
        }
    })

if __name__ == '__main__':
    # Crear directorios
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("🚀 PaddleOCR 3.0.2 Server iniciando...")
    print(f"📁 Input: {UPLOAD_FOLDER}")
    print(f"📁 Output: {OUTPUT_FOLDER}")
    print("💡 Usa /init para precargar modelos")
    print("🔥 PaddleOCR 3.0.2 + PaddlePaddle 3.0.0")
    print("🎯 Sintaxis CORRECTA: .predict() API")
    print("✅ Extracción de texto optimizada")
    
    app.run(host='0.0.0.0', port=8501, debug=False)
