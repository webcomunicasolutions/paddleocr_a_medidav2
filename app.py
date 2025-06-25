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
    """Cálculo inteligente de side_len como tu amigo"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return 960
        
        h, w = img.shape[:2]
        side_len = int(math.ceil(max(h, w) * max(0.8, 960 / max(h, w))))
        print(f"📐 Imagen {w}x{h} -> side_len: {side_len}px")
        return side_len
    except:
        return 960

def initialize_ocr():
    global ocr_instances, ocr_initialized
    
    if ocr_initialized:
        return True
    
    try:
        print("🚀 Inicializando PaddleOCR 3.0.2...")
        from paddleocr import PaddleOCR
        
        # Configuración SIMPLE que funciona
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

def detect_text_orientation_improved(coordinates):
    """Detección mejorada de orientación"""
    try:
        if len(coordinates) >= 4:
            x_coords = [point[0] for point in coordinates]
            y_coords = [point[1] for point in coordinates]
            
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            
            if width == 0:
                return 'vertical'
            
            aspect_ratio = height / width
            p1, p2 = coordinates[0], coordinates[1]
            angle = abs(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi)
            
            if aspect_ratio > 2.5:
                return 'vertical'
            elif angle > 25 and angle < 155:
                return 'rotated'
            elif aspect_ratio > 1.8:
                return 'vertical'
            else:
                return 'horizontal'
    except:
        pass
    return 'horizontal'

def analyze_text_orientations(coordinates_list):
    """Análisis de orientaciones"""
    orientations = {'horizontal': 0, 'vertical': 0, 'rotated': 0}
    
    for coords in coordinates_list:
        orientation = detect_text_orientation_improved(coords)
        orientations[orientation] += 1
    
    return orientations

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
                # NUEVO: OCR directo con .predict() (funciona para PDF e imágenes)
                print(f"🔍 Procesando {filename} con PaddleOCR 3.0.2...")
                result = ocr.predict(tmp_file.name)
                print(f"✅ OCR completado")
                
            finally:
                os.remove(tmp_file.name)
        
        # Extraer datos del resultado usando la estructura que funciona
        if not result or len(result) == 0:
            return jsonify({
                'success': False,
                'error': 'No OCR result',
                'processing_time': time.time() - start_time
            })
        
        page_result = result[0]
        text_lines = page_result.get('rec_texts', [])
        confidences = page_result.get('rec_scores', [])
        coordinates_list = page_result.get('dt_polys', [])
        
        # Analizar orientaciones
        orientations = analyze_text_orientations(coordinates_list)
        
        # Estadísticas
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
            'has_coordinates': len(coordinates_list) > 0,
            'text_orientations': orientations,
            'has_vertical_text': orientations.get('vertical', 0) > 0,
            'has_rotated_text': orientations.get('rotated', 0) > 0,
            'ocr_version': '3.0.2',
            'pdf_support': 'native'
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
                    if hasattr(coords, 'tolist'):
                        coords = coords.tolist()
                    block_info['coordinates'] = coords
                    block_info['orientation'] = detect_text_orientation_improved(coords)
                
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
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'processing_time': round(processing_time, 3)
        }), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("🚀 PaddleOCR 3.0.2 Server iniciando...")
    print("🔄 Pre-cargando modelos OCR...")
    
    if initialize_ocr():
        print("✅ Modelos pre-cargados exitosamente")
        print("🎯 Las siguientes peticiones serán instantáneas")
    else:
        print("⚠️ Error pre-cargando modelos")
    
    print("🌐 Servidor listo en puerto 8501")
    print("📍 Funcionalidades:")
    print("   ✅ PaddleOCR 3.0.2 nativo")
    print("   ✅ Soporte PDF directo")
    print("   ✅ Método .predict() estable")
    print("   ✅ Detección de orientación")
    print("   ✅ Coordenadas y confianza")
    
    app.run(host='0.0.0.0', port=8501, debug=False)
