#!/usr/bin/env python3
import os
import json
import time
import tempfile
import numpy as np
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
        print("🚀 Inicializando PaddleOCR con detección de orientación...")
        from paddleocr import PaddleOCR
        
        # Configuración mejorada para detectar texto vertical
        # use_angle_cls=True habilita detección de orientación (0°, 90°, 180°, 270°)
        ocr_instances["es"] = PaddleOCR(lang='es', use_angle_cls=True)
        ocr_instances["en"] = PaddleOCR(lang='en', use_angle_cls=True)
        
        ocr_initialized = True
        print("✅ OCR inicializado con detección de orientación")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def detect_text_orientation(coordinates):
    """
    Detectar orientación del texto basándose en coordenadas
    Retorna: 'horizontal', 'vertical', 'rotated'
    """
    try:
        if len(coordinates) >= 4:
            # Calcular dimensiones del rectángulo
            x_coords = [point[0] for point in coordinates]
            y_coords = [point[1] for point in coordinates]
            
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            
            # Calcular ángulo de rotación
            p1, p2 = coordinates[0], coordinates[1]
            angle = abs(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi)
            
            # Clasificar orientación
            if height > (width * 1.5):  # Texto claramente vertical
                return 'vertical'
            elif angle > 15 and angle < 165:  # Texto rotado
                return 'rotated'
            else:
                return 'horizontal'
    except:
        pass
    return 'horizontal'

def analyze_text_orientations(coordinates_list):
    """
    Analizar todas las orientaciones de texto en el documento
    """
    orientations = {'horizontal': 0, 'vertical': 0, 'rotated': 0}
    
    for coords in coordinates_list:
        orientation = detect_text_orientation(coords)
        orientations[orientation] += 1
    
    return orientations

def get_ocr_instance(language=None):
    global ocr_instances, ocr_initialized
    
    if not ocr_initialized:
        if not initialize_ocr():
            return None
    
    lang = language or default_lang
    return ocr_instances.get(lang, ocr_instances.get("es"))
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
                    result = ocr.predict(img_tmp.name)
                    os.remove(img_tmp.name)
            else:
                result = ocr.predict(tmp_file.name)
            
            os.remove(tmp_file.name)
        
        # Extraer texto (método que funcionaba)
        text_lines = []
        if result and isinstance(result, list) and len(result) > 0:
            page_result = result[0]
            if 'rec_texts' in page_result:
                text_lines = page_result['rec_texts']
        
        # Extraer confianzas
        confidences = []
        if result and isinstance(result, list) and len(result) > 0:
            page_result = result[0]
            if 'rec_scores' in page_result:
                confidences = page_result['rec_scores']
        
        # NUEVO: Extraer coordenadas básicas
        coordinates_list = []
        if result and isinstance(result, list) and len(result) > 0:
            page_result = result[0]
            if 'dt_polys' in page_result:
                coordinates_list = page_result['dt_polys']
        
        # NUEVO: Analizar orientaciones de texto
        orientations = analyze_text_orientations(coordinates_list)
        
        # Calcular estadísticas
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Respuesta básica
        response = {
            'success': True,
            'text': '\n'.join(text_lines),
            'total_blocks': len(text_lines),
            'filename': filename,
            'language': language,
            'avg_confidence': round(avg_confidence, 3) if avg_confidence > 0 else None,
            'has_coordinates': len(coordinates_list) > 0,
            'text_orientations': orientations,
            'has_vertical_text': orientations.get('vertical', 0) > 0,
            'has_rotated_text': orientations.get('rotated', 0) > 0
        }
        
        # Si piden detalle, agregar coordenadas y más info
        if detailed:
            blocks_with_coords = []
            for i, text in enumerate(text_lines):
                block_info = {'text': text}
                
                if i < len(confidences):
                    block_info['confidence'] = round(confidences[i], 3)
                
                if i < len(coordinates_list):
                    coords = coordinates_list[i].tolist() if hasattr(coordinates_list[i], 'tolist') else coordinates_list[i]
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
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("🚀 PaddleOCR Server iniciando...")
    print("🔄 Pre-cargando modelos OCR (primera vez puede tardar 2-3 minutos)...")
    
    # OPCIÓN 1: Pre-cargar modelos al arrancar
    if initialize_ocr():
        print("✅ Modelos OCR pre-cargados exitosamente")
        print("🎯 Las siguientes peticiones serán instantáneas")
    else:
        print("⚠️ Error pre-cargando modelos, se cargarán en primera petición")
    
    print("🌐 Servidor listo en puerto 8501")
    print("📍 Funcionalidades:")
    print("   ✅ Texto extraído")
    print("   ✅ Coordenadas disponibles")
    print("   ✅ Confianza/probabilidades")
    print("   ✅ Detección de texto vertical")
    print("   ✅ Detección de texto rotado")
    print("   ✅ Modo detailed con parámetro 'detailed=true'")
    
    app.run(host='0.0.0.0', port=8501, debug=False)
