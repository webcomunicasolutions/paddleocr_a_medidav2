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

def debug_ocr_result(ocr_result):
    """
    DEBUG: Imprimir la estructura exacta que devuelve OCR
    """
    print("\n" + "="*50)
    print("🔍 DEBUG - Estructura OCR:")
    print("="*50)
    
    print(f"Tipo de resultado: {type(ocr_result)}")
    print(f"Es lista?: {isinstance(ocr_result, list)}")
    
    if ocr_result:
        print(f"Longitud: {len(ocr_result)}")
        
        for i, item in enumerate(ocr_result):
            print(f"\nItem {i}:")
            print(f"  Tipo: {type(item)}")
            print(f"  Es lista?: {isinstance(item, list)}")
            
            if isinstance(item, list):
                print(f"  Longitud: {len(item)}")
                
                for j, subitem in enumerate(item[:3]):  # Solo primeros 3
                    print(f"    SubItem {j}:")
                    print(f"      Tipo: {type(subitem)}")
                    
                    if isinstance(subitem, list) and len(subitem) >= 2:
                        print(f"      Longitud: {len(subitem)}")
                        print(f"      Elem 0 (coords): {type(subitem[0])}")
                        print(f"      Elem 1 (texto): {type(subitem[1])}")
                        
                        if len(subitem) > 1:
                            texto_parte = subitem[1]
                            print(f"      Texto parte tipo: {type(texto_parte)}")
                            if isinstance(texto_parte, (list, tuple)):
                                print(f"      Texto contenido: {texto_parte}")
                            else:
                                print(f"      Texto valor: {str(texto_parte)[:50]}")
                
                if len(item) > 3:
                    print(f"    ... y {len(item)-3} elementos más")
    
    print("="*50 + "\n")
    return ocr_result

def extract_text_simple(ocr_result):
    """
    Extracción SIMPLE - probamos diferentes estructuras
    """
    all_text = []
    
    if not ocr_result:
        return "Sin resultado OCR"
    
    try:
        # Estructura 1: Lista de páginas > lista de bloques > [coords, (texto, conf)]
        if isinstance(ocr_result, list):
            for page in ocr_result:
                if isinstance(page, list):
                    for block in page:
                        if isinstance(block, list) and len(block) >= 2:
                            texto_parte = block[1]
                            
                            # Caso A: (texto, confianza)
                            if isinstance(texto_parte, (list, tuple)) and len(texto_parte) >= 1:
                                texto = str(texto_parte[0]).strip()
                                if texto:
                                    all_text.append(texto)
                            
                            # Caso B: solo texto
                            elif isinstance(texto_parte, str):
                                texto = texto_parte.strip()
                                if texto:
                                    all_text.append(texto)
        
        # Si no encontramos nada, intentar estructura alternativa
        if not all_text:
            # Estructura 2: Intentar acceso directo
            for item in ocr_result:
                if isinstance(item, str):
                    all_text.append(item)
                elif hasattr(item, 'text'):
                    all_text.append(str(item.text))
    
    except Exception as e:
        print(f"❌ Error extrayendo texto: {e}")
        return f"Error: {str(e)}"
    
    if all_text:
        return '\n'.join(all_text)
    else:
        return "No se pudo extraer texto - revisar estructura"

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
                    result = ocr.ocr(img_tmp.name)
                    os.remove(img_tmp.name)
            else:
                result = ocr.ocr(tmp_file.name)
            
            os.remove(tmp_file.name)
        
        # DEBUG: Mostrar estructura en consola
        debug_ocr_result(result)
        
        # Extraer texto con método simple
        texto_extraido = extract_text_simple(result)
        
        return jsonify({
            'success': True,
            'text': texto_extraido,
            'filename': filename,
            'language': language,
            'raw_result_type': str(type(result)),
            'raw_result_length': len(result) if result else 0,
            'debug_info': 'Revisa la consola del servidor para ver la estructura completa'
        })
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Error completo:\n{error_detail}")
        return jsonify({'error': str(e), 'detail': error_detail}), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print("🚀 PaddleOCR DEBUG Server iniciando...")
    print("🔍 Esta versión mostrará la estructura exacta del resultado OCR")
    app.run(host='0.0.0.0', port=8501, debug=False)
