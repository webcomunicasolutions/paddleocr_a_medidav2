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
    """Inicializa PaddleOCR 3.0 con configuración específica"""
    global ocr_instances, ocr_initialized
    
    if ocr_initialized:
        return True
    
    try:
        print("🚀 Inicializando PaddleOCR 3.0...")
        
        # Importar después de que todo esté instalado
        import paddle
        from paddleocr import PaddleOCR
        
        # Configuración específica para PaddleOCR 3.0
        paddle_config = {
            "use_gpu": False,
            "enable_mkldnn": False,  # Deshabilitar MKL-DNN para evitar conflictos
            "cpu_threads": 1,
            "use_tensorrt": False,
            "precision": "fp32"
        }
        
        # Configurar PaddlePaddle
        paddle.set_device('cpu')
        
        for lang in supported_languages:
            print(f"📚 Cargando modelo para {lang.upper()}...")
            
            ocr_config = {
                "lang": lang,
                "use_gpu": False,
                "show_log": False,
                "use_angle_cls": True,
                "use_space_char": True,
                "det_limit_side_len": 960,
                "det_limit_type": 'max',
                "rec_batch_num": 6
            }
            
            # Crear instancia OCR
            ocr_instances[lang] = PaddleOCR(**ocr_config)
            print(f"✅ OCR {lang.upper()} inicializado")
        
        ocr_initialized = True
        print("🎉 PaddleOCR 3.0 completamente inicializado")
        return True
        
    except Exception as e:
        print(f"❌ Error inicializando PaddleOCR 3.0: {e}")
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
    
    return ocr_instances.get(lang)

@app.route('/health')
def health():
    """Health check que no requiere OCR pesado"""
    return jsonify({
        'status': 'healthy',
        'service': 'PaddleOCR 3.0 Server',
        'languages': supported_languages,
        'ocr_ready': ocr_initialized,
        'version': '3.0.1'
    })

@app.route('/init')
def init_models():
    """Endpoint para inicializar modelos manualmente"""
    try:
        success = initialize_ocr()
        return jsonify({
            'success': success,
            'models_loaded': list(ocr_instances.keys()) if success else [],
            'message': 'Models initialized successfully' if success else 'Failed to initialize models'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return jsonify({
        'service': 'PaddleOCR 3.0 Server',
        'version': '3.0.1',
        'pdf_support': True,
        'endpoints': {
            'health': '/health',
            'init': '/init (initialize models)',
            'process': '/process (POST)',
            'status': '/status'
        },
        'languages': supported_languages,
        'features': ['Multi-page PDF', 'Advanced layout analysis', 'Enhanced text detection']
    })

@app.route('/process', methods=['POST'])
def process_file():
    """Procesar archivo con PaddleOCR 3.0"""
    start_time = time.time()
    temp_files = []
    
    try:
        # Inicializar OCR si no está listo
        if not ocr_initialized:
            init_success = initialize_ocr()
            if not init_success:
                return jsonify({'error': 'OCR not initialized'}), 503
        
        # Validar archivo
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        language = request.form.get('language', default_lang)
        process_pages = int(request.form.get('pages', 1))  # Número de páginas PDF
        
        # Obtener OCR
        ocr = get_ocr_instance(language)
        if ocr is None:
            return jsonify({'error': f'OCR {language} not available'}), 503
        
        # Guardar archivo
        filename = secure_filename(file.filename)
        
        # Usar archivo temporal para evitar problemas de permisos
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
            file.save(tmp_file.name)
            temp_files.append(tmp_file.name)
            process_path = tmp_file.name
        
        # Procesar según el tipo de archivo
        if filename.lower().endswith('.pdf'):
            print(f"📄 Procesando PDF: {filename} ({process_pages} páginas)")
            
            # PaddleOCR 3.0 puede manejar PDFs directamente
            try:
                # Procesamiento directo de PDF con PaddleOCR 3.0
                result = ocr.ocr(process_path, cls=True)
                pdf_processed = True
            except Exception as pdf_error:
                print(f"⚠️ Error procesando PDF directamente: {pdf_error}")
                print("🔄 Convirtiendo PDF a imagen como fallback...")
                
                # Fallback: convertir a imagen
                from pdf2image import convert_from_path
                pages = convert_from_path(
                    process_path, 
                    dpi=300, 
                    first_page=1, 
                    last_page=min(process_pages, 3)  # Máximo 3 páginas
                )
                
                if not pages:
                    return jsonify({'error': 'No pages found in PDF'}), 400
                
                # Procesar primera página como imagen
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as img_tmp:
                    pages[0].save(img_tmp.name, 'JPEG', quality=95)
                    temp_files.append(img_tmp.name)
                    result = ocr.ocr(img_tmp.name, cls=True)
                    pdf_processed = False
        else:
            print(f"🖼️ Procesando imagen: {filename}")
            result = ocr.ocr(process_path, cls=True)
            pdf_processed = False
        
        # Extraer texto
        all_text = []
        total_blocks = 0
        
        if result:
            if isinstance(result, list) and len(result) > 0:
                # PaddleOCR 3.0 puede devolver resultados de múltiples páginas
                for page_result in result:
                    if page_result:
                        for line in page_result:
                            if len(line) >= 2 and line[1]:
                                all_text.append(line[1][0])
                                total_blocks += 1
        
        extracted_text = '\n'.join(all_text)
        processing_time = time.time() - start_time
        
        # Respuesta
        response = {
            'success': True,
            'text': extracted_text,
            'language': language,
            'filename': filename,
            'processing_time': round(processing_time, 3),
            'text_blocks_found': total_blocks,
            'pdf_direct_processing': pdf_processed if filename.lower().endswith('.pdf') else None,
            'paddleocr_version': '3.0.1',
            'timestamp': time.time()
        }
        
        print(f"✅ Procesado en {processing_time:.2f}s - {total_blocks} bloques de texto")
        return jsonify(response)
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Error procesando: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e),
            'processing_time': round(processing_time, 3)
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
        'service': 'PaddleOCR 3.0 Server',
        'version': '3.0.1',
        'languages': supported_languages,
        'formats': list(ALLOWED_EXTENSIONS),
        'models_loaded': list(ocr_instances.keys()),
        'ocr_initialized': ocr_initialized,
        'features': {
            'pdf_direct_processing': True,
            'multi_page_support': True,
            'layout_analysis': True
        }
    })

if __name__ == '__main__':
    # Crear directorios
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("🚀 PaddleOCR 3.0 Server iniciando...")
    print(f"📁 Input: {UPLOAD_FOLDER}")
    print(f"📁 Output: {OUTPUT_FOLDER}")
    print("💡 Usa /init para precargar modelos")
    print("📄 Soporte nativo de PDF habilitado")
    
    app.run(host='0.0.0.0', port=8501, debug=False)
