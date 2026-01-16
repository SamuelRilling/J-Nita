"""
Flask backend API for J-Nita v5.0 web application
Handles image conditioning and OCR processing
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import base64
import io
import tempfile
from PIL import Image
import numpy as np
import cv2
from image_conditioner import ImageConditioner
from gradio_client import Client, handle_file
import logging

app = Flask(__name__)
# Enable CORS for all origins (required for GitHub Pages)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global OCR client (lazy initialization)
ocr_client = None

def get_ocr_client():
    """Initialize OCR client if not already done."""
    global ocr_client
    if ocr_client is None:
        try:
            ocr_client = Client("prithivMLmods/Multimodal-OCR3")
            logger.info("OCR client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OCR client: {e}")
            raise
    return ocr_client


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route('/api/condition', methods=['POST'])
def condition_image():
    """
    Condition an image for OCR.
    Expects: JSON with 'image' (base64), 'config' (optional config overrides)
    Returns: Base64 encoded conditioned image
    """
    try:
        data = request.json
        image_data = data.get('image')
        config_overrides = data.get('config', {})
        
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Load default config
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Apply config overrides
        ic_config = config.get('image_conditioning', {})
        # Handle config overrides (can be nested or flat)
        for key, value in config_overrides.items():
            if key in ic_config:
                if isinstance(ic_config[key], dict) and 'value' in ic_config[key]:
                    ic_config[key]['value'] = value
                else:
                    ic_config[key] = value
        
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_input:
            cv2.imwrite(tmp_input.name, image_array)
            input_path = tmp_input.name
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmp_output:
            # Initialize conditioner
            conditioner = ImageConditioner(
                input_folder=os.path.dirname(input_path),
                output_folder=tmp_output,
                strength=ic_config.get('strength', {}).get('value', 10),
                adaptive_block_size=ic_config.get('adaptive_block_size', {}).get('value'),
                adaptive_C=ic_config.get('adaptive_C', {}).get('value'),
                morph_kernel_size=tuple(ic_config.get('morph_kernel_size', {}).get('value', [2, 2])) if ic_config.get('morph_kernel_size', {}).get('value') else None,
                morph_iterations=ic_config.get('morph_iterations', {}).get('value'),
                target_width=ic_config.get('target_width', {}).get('value'),
                target_height=ic_config.get('target_height', {}).get('value'),
                canny_threshold1=ic_config.get('canny_threshold1', {}).get('value'),
                canny_threshold2=ic_config.get('canny_threshold2', {}).get('value'),
                min_contour_area=ic_config.get('min_contour_area', {}).get('value'),
                max_pages=ic_config.get('max_pages', {}).get('value'),
                denoise_ksize=ic_config.get('denoise_ksize', {}).get('value'),
                png_compression=ic_config.get('png_compression', {}).get('value', 0),
                min_scale_for_zero=ic_config.get('min_scale_for_zero', {}).get('value'),
                debug_mode=False
            )
            
            # Process image
            filename = os.path.basename(input_path)
            conditioner._process_single_image(input_path, filename)
            
            # Find output file
            output_files = [f for f in os.listdir(tmp_output) if f.endswith('.png')]
            if not output_files:
                return jsonify({"error": "No output image generated"}), 500
            
            output_path = os.path.join(tmp_output, output_files[0])
            
            # Read and encode output image
            output_image = cv2.imread(output_path)
            _, buffer = cv2.imencode('.png', output_image)
            output_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Cleanup
            os.unlink(input_path)
            
            return jsonify({
                "conditioned_image": f"data:image/png;base64,{output_base64}",
                "success": True
            })
    
    except Exception as e:
        logger.error(f"Error conditioning image: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/ocr', methods=['POST'])
def process_ocr():
    """
    Process OCR on a conditioned image.
    Expects: JSON with 'image' (base64), 'model' (optional), 'fallback_models' (optional)
    Returns: OCR text results
    """
    try:
        data = request.json
        image_data = data.get('image')
        model = data.get('model', 'Nanonets-OCR2-3B')
        fallback_models = data.get('fallback_models', [])
        enable_fallback = data.get('enable_fallback', True)
        
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name
        
        try:
            client = get_ocr_client()
            
            # Build list of models to try
            models_to_try = [model]
            if enable_fallback and fallback_models:
                models_to_try.extend([m for m in fallback_models if m != model])
            
            # Try models in order
            raw_output = ""
            markdown_output = ""
            model_used = None
            last_error = ""
            
            for model_name in models_to_try:
                try:
                    logger.info(f"Trying OCR model: {model_name}")
                    result = client.predict(
                        model_name,
                        "Perform OCR on the image.",
                        handle_file(tmp_path),
                        2048,
                        0.7,
                        0.9,
                        50,
                        1.1,
                        api_name="/generate_image"
                    )
                    
                    raw_output = str(result[0]) if isinstance(result, (list, tuple)) and len(result) > 0 else ""
                    markdown_output = str(result[1]) if isinstance(result, (list, tuple)) and len(result) > 1 else raw_output
                    model_used = model_name
                    logger.info(f"Success with model: {model_name}")
                    break
                    
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Failed with {model_name}: {last_error}")
                    if "timeout" not in last_error.lower() and "gpu" not in last_error.lower():
                        break
            
            if not raw_output and not markdown_output:
                return jsonify({
                    "error": f"All models failed. Last error: {last_error}",
                    "success": False
                }), 500
            
            return jsonify({
                "raw_text": raw_output,
                "markdown_text": markdown_output,
                "model_used": model_used,
                "success": True
            })
        
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"Error processing OCR: {e}", exc_info=True)
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        return jsonify(config)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/config', methods=['POST'])
def save_config():
    """Save configuration."""
    try:
        config = request.json
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
