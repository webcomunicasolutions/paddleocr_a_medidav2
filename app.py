#!/usr/bin/env python3
import os
import json
import time
import tempfile
import numpy as np
import cv2
import math
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

def calculate_intelligent_side_len(image_path):
    """
    Cálculo inteligente de side_len como hace tu amigo
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return 960  # fallback
        
        h, w = img.shape[:2]
        side_len = int(math.ceil(max(h, w) * max(0.8, 960 / max(h, w))))
        print(f"📐 Imagen {w}x{h} -> side_len calculado: {side_len}px")
        return side_len
    except:
        return 960

def initialize_ocr():
    global ocr_instances, ocr_initialized
    
    if ocr_initialized:
        return True
    
    try:
        print("🚀 Inicializando PaddleOCR v4 con configuración avanzada...")
        from paddleocr import PaddleOCR
        
        # Configuración para ESPAÑOL (PP-OCRv4)
        print("📚 Cargando OCR para ESPAÑOL...")
        ocr_instances["es"] = PaddleOCR(
            ocr_version='PP-OCRv4',
            det_model_dir='en_PP-OCRv4_server_det',        # Detector v4
            rec_model_dir='es_PP-OCRv4_server_rec',        # Reconocedor español v4
            cls_model_dir='ch_ppocr_server_v2.0_cls_infer', # Clasificador de ángulos
            lang='es',
            use_angle_cls=True,                            # Detección de ángulos
            use_textline_orientation=True,                 # Orientación de líneas
            use_doc_orientation_classify=False,            # No clasificar documento completo
            use_doc_unwarping=False,                       # No enderezar documento
            det_db_box_thresh=0.3,                        # Umbral de detección de cajas
            det_db_thresh=0.25,                           # Umbral de segmentación
            enable_mkldnn=True,                           # Optimización CPU
            use_gpu=False,                                # CPU por ahora
            show_log=False                                # Sin logs verbosos
        )
        
        # Configuración para INGLÉS (PP-OCRv4)
        print("📚 Cargando OCR para INGLÉS...")
        ocr_instances["en"] = PaddleOCR(
            ocr_version='PP-OCRv4',
            det_model_dir='en_PP-OCRv4_server_det',
            rec_model_dir='en_PP-OCRv4_server_rec',        # Reconocedor inglés v4
            cls_model_dir='ch_ppocr_server_v2.0_cls_infer',
            lang='en',
            use_angle_cls=True,
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            det_db_box_thresh=0.3,
            det_db_thresh=0.25,
            enable_mkldnn=True,
            use_gpu=False,
            show_log=False
        )
        
        ocr_initialized = True
        print("✅ OCR v4 inicializado con modelos avanzados")
        
        # Mostrar modelos cargados como hace tu amigo
        for lang, ocr_instance in ocr_instances.items():
            args = ocr_instance.args
            print(f"""
🔧 Modelos cargados para {lang.upper()}:
─────────────────────────────────────────
  Detector   : {args.det_model_dir}
  Recognizer : {args.rec_model_dir}
  Classifier : {args.cls_model_dir}
─────────────────────────────────────────""")
        
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

def detect_text_orientation(coordinates):
    """
    Detectar orientación del texto (mejorado)
    """
    try:
        if len(coordinates) >= 4:
            x_coords = [point[0] for point in coordinates]
            y_coords = [point[1] for point in coordinates]
            
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            
            # Cálculo de ángulo más preciso
            p1, p2 = coordinates[0], coordinates[1]
            angle = abs(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi)
            
            # Lógica mejorada de clasificación
            aspect_ratio = height / width if width > 0 else 0
            
            if aspect_ratio > 2.0:  # Muy vertical
                return 'vertical'
            elif angle > 20 and angle < 160:  # Claramente rotado
                return 'rotated'
            elif aspect_ratio > 1.5:  # Posiblemente vertical
                return 'vertical'
            else:
                return 'horizontal'
    except:
        pass
    return 'horizontal'

def analyze_text_orientations(coordinates_list):
    """
    Analizar orientaciones con mejor precisión
    """
    orientations = {'horizontal': 0, 'vertical': 0, 'rotated': 0}
    
    for coords in coordinates_list:
        orientation = detect_text_orientation(coords)
        orientations[orientation] += 1
    
    return orientations

def process_ocr_result_v4(ocr_result):
    """
    Procesar resultado de OCR v4 con el formato mejorado
    """
    text_lines = []
    confidences = []
    coordinates_list = []
    
    if not ocr_result or not isinstance(ocr_result, list):
        return text_lines, confidences, coordinates_list
    
    try:
        for page_result in ocr_result:
            if not page_result:
                continue
                
            for word_info in page_result:
                try:
                    # Formato OCR v4: [coordenadas, (texto, confianza)]
                    if len(word_info) >= 2:
                        coordinates = word_info[0]
                        text_data = word_info[1]
                        
                        if isinstance(text_data, (list, tuple)) and len(text_data) >= 2:
                            text = str(text_data[0]).strip()
                            confidence = float(text_data[1])
                            
                            if text:  # Solo agregar si hay texto
                                text_lines.append(text)
                                confidences.append(confidence)
                                coordinates_list.append(coordinates)
                                
                except Exception as e:
                    print(f"⚠️ Error procesando palabra: {e}")
                    continue
                    
    except Exception as e:
        print(f"⚠️ Error procesando resultado OCR: {e}")
    
    return text_lines, confidences, coordinates_list

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'ocr_ready': ocr_initialized, 'version': 'PP-OCRv4'})

@app.route('/init')
def init_models():
    try:
        success = initialize_ocr()
        return jsonify({
            'success': success,
            'models_loaded': list(ocr_instances.keys()) if success else [],
            'version': 'PP-OCRv4'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_file():
    start_time = time.time()
    
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
            
            try:
                if filename.lower().endswith('.pdf'):
                    from pdf2image import convert_from_path
                    pages = convert_from_path(tmp_file.name, dpi=300, first_page=1, last_page=1)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as img_tmp:
                        pages[0].save(img_tmp.name, 'JPEG', quality=95)
                        
                        # Calcular side_len inteligente
                        side_len = calculate_intelligent_side_len(img_tmp.name)
                        
                        # Actualizar configuración dinámicamente
                        ocr.args.det_limit_side_len = side_len
                        
                        # OCR v4 con cls=True
                        result = ocr.ocr(img_tmp.name, cls=True)
                        os.remove(img_tmp.name)
                else:
                    # Calcular side_len para imagen
                    side_len = calculate_intelligent_side_len(tmp_file.name)
                    ocr.args.det_limit_side_len = side_len
                    
                    result = ocr.ocr(tmp_file.name, cls=True)
                
            finally:
                os.remove(tmp_file.name)
        
        # Procesar resultado con nuevo formato
        text_lines, confidences, coordinates_list = process_ocr_result_v4(result)
        
        # Analizar orientaciones
        orientations = analyze_text_orientations(coordinates_list)
        
        # Calcular estadísticas
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        processing_time = time.time() - start_time
        
        # Respuesta básica
        response = {
            'success': True,
            'text': '\n'.join(text_lines),
            'total_blocks': len(text_lines),
            'filename': filename,
            'language': language,
            'avg_confidence': round(avg_confidence, 3) if avg_confidence > 0 else None,
            'processing_time': round(processing_time, 3),
            'ocr_version': 'PP-OCRv4',
            'has_coordinates': len(coordinates_list) > 0,
            'text_orientations': orientations,
            'has_vertical_text': orientations.get('vertical', 0) > 0,
            'has_rotated_text': orientations.get('rotated', 0) > 0
        }
        
        # Modo detallado
        if detailed:
            blocks_with_coords = []
            for i, text in enumerate(text_lines):
                block_info = {'text': text}
                
                if i < len(confidences):
                    block_info['confidence'] = round(confidences[i], 3)
                
                if i < len(coordinates_list):
                    coords = coordinates_list[i]
                    block_info['coordinates'] = coords
                    block_info['orientation'] = detect_text_orientation(coords)
                
                blocks_with_coords.append(block_info)
            
            response.update({
                'blocks': blocks_with_coords,
                'min_confidence': round(min(confidences), 3) if confidences else None,
                'max_confidence': round(max(confidences), 3) if confidences else None,
                'total_coordinates': len(coordinates_list),
                'orientation_details': {
                    'horizontal_blocks': orientations.get('horizontal', 0),
                    'vertical_blocks': orientations.get('vertical', 0),
                    'rotated_blocks': orientations.get('rotated', 0)
                }
            })
        
        return jsonify(response)
        
    except Exception as e:
        processing_time = time.time() - start_time
        return jsonify({
            'error': str(e),
            'processing_time': round(processing_time, 3)
        }), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("🚀 PaddleOCR v4 Server iniciando...")
    print("🔄 Pre-cargando modelos OCR v4 (primera vez puede tardar 3-5 minutos)...")
    
    # Pre-cargar modelos al arrancar
    if initialize_ocr():
        print("✅ Modelos OCR v4 pre-cargados exitosamente")
        print("🎯 Las siguientes peticiones serán instantáneas")
    else:
        print("⚠️ Error pre-cargando modelos, se cargarán en primera petición")
    
    print("🌐 Servidor listo en puerto 8501")
    print("📍 Funcionalidades PP-OCRv4:")
    print("   ✅ Modelos v4 más precisos")
    print("   ✅ Side_len inteligente automático")
    print("   ✅ Detección de orientación mejorada")
    print("   ✅ Coordenadas exactas")
    print("   ✅ Confianza por bloque")
    print("   ✅ Optimización CPU (MKLDNN)")
    
    app.run(host='0.0.0.0', port=8501, debug=False)
