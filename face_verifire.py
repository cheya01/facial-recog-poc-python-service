from flask import Flask, request, jsonify
from deepface import DeepFace
import logging
from datetime import datetime
import os
import tempfile
import cv2
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_verification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def preprocess_image(image_path, request_id):
    """
    Preprocess image to improve face detection for webcam/low-quality images
    """
    try:
        logger.info(f"[{request_id}] Preprocessing image: {image_path}")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"[{request_id}] Could not read image with OpenCV, trying PIL")
            # Fallback to PIL
            pil_img = Image.open(image_path)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        original_shape = img.shape
        logger.info(f"[{request_id}] Original image shape: {original_shape}")
        
        # 1. Convert to grayscale for processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply histogram equalization to improve contrast
        equalized = cv2.equalizeHist(gray)
        
        # 3. Apply slight Gaussian blur to reduce noise
        denoised = cv2.GaussianBlur(equalized, (3, 3), 0)
        
        # 4. Convert back to BGR
        processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        
        # 5. Resize if image is too small (minimum 640px on shorter side)
        height, width = processed.shape[:2]
        min_dimension = min(height, width)
        
        if min_dimension < 640:
            scale_factor = 640 / min_dimension
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            logger.info(f"[{request_id}] Upscaled image from {width}x{height} to {new_width}x{new_height}")
        
        # 6. Apply sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(processed, -1, kernel)
        
        # 7. Save preprocessed image
        preprocessed_path = image_path.replace('.jpg', '_preprocessed.jpg').replace('.png', '_preprocessed.png')
        cv2.imwrite(preprocessed_path, sharpened, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        logger.info(f"[{request_id}] Preprocessed image saved to: {preprocessed_path}")
        return preprocessed_path
        
    except Exception as e:
        logger.error(f"[{request_id}] Error preprocessing image: {str(e)}", exc_info=True)
        # Return original path if preprocessing fails
        return image_path

@app.route('/compare', methods=['POST'])
def compare_faces():
    request_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    logger.info(f"[{request_id}] Received face comparison request")
    
    # Temporary file paths
    img1_path = None
    img2_path = None
    
    try:
        # Validate request files
        if 'image1' not in request.files or 'image2' not in request.files:
            logger.warning(f"[{request_id}] Missing image files in request")
            return jsonify({'error': 'Both image1 and image2 are required'}), 400
        
        logger.info(f"[{request_id}] Saving uploaded images to temporary files")
        
        # Save uploaded files to temporary location
        img1_file = request.files['image1']
        img2_file = request.files['image2']
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(img1_file.filename)[1]) as tmp1:
            img1_path = tmp1.name
            img1_file.save(img1_path)
            logger.info(f"[{request_id}] Image1 saved to {img1_path}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(img2_file.filename)[1]) as tmp2:
            img2_path = tmp2.name
            img2_file.save(img2_path)
            logger.info(f"[{request_id}] Image2 saved to {img2_path}")

        # Preprocess images to improve face detection for webcam/low-quality images
        img1_processed = preprocess_image(img1_path, request_id)
        img2_processed = preprocess_image(img2_path, request_id)

        logger.info(f"[{request_id}] Running DeepFace verification")
        
        # Try multiple detector backends for better face detection
        detectors = ['retinaface', 'mtcnn', 'ssd', 'opencv']  # Reordered: most accurate first
        result = None
        last_error = None
        
        for detector in detectors:
            try:
                logger.info(f"[{request_id}] Trying detector: {detector}")
                result = DeepFace.verify(
                    img1_path=img1_processed,
                    img2_path=img2_processed,
                    model_name='VGG-Face',  # Options: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace, Dlib
                    detector_backend=detector,
                    enforce_detection=True,  # Enforce face detection to avoid false results
                    align=True
                )
                
                # Check if faces were actually detected
                facial_areas = result.get('facial_areas', {})
                img1_area = facial_areas.get('img1', {})
                img2_area = facial_areas.get('img2', {})
                
                # Verify that both images have valid face detections
                if (img1_area.get('x') is not None and img2_area.get('x') is not None):
                    logger.info(f"[{request_id}] Successfully used detector: {detector} - Faces detected in both images")
                    break
                else:
                    logger.warning(f"[{request_id}] Detector '{detector}' succeeded but faces not properly detected")
                    result = None
                    last_error = "Faces not detected in one or both images"
                    continue
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[{request_id}] Detector '{detector}' failed: {str(e)}")
                result = None
                continue
        
        if result is None:
            raise ValueError(f"Could not detect faces in both images. Last error: {last_error}")
        
        logger.info(f"[{request_id}] DeepFace result: {result}")
        
        # Check for infinity distance (indicates failed face detection despite success)
        distance = result['distance']
        if distance == float('inf'):
            raise ValueError("Face detection returned infinite distance - faces likely not detected properly")
        
        # Extract results
        match = result['verified']
        threshold = result['threshold']
        model = result['model']
        detector_used = result.get('detector_backend', 'unknown')
        
        # Calculate confidence (inverse of distance, normalized)
        # Lower distance means higher confidence
        confidence = max(0, (1 - (distance / threshold)) * 100) if threshold > 0 else 0

        logger.info(f"[{request_id}] Comparison complete - Match: {match}, Confidence: {round(confidence, 2)}%, Distance: {distance:.4f}, Threshold: {threshold:.4f}, Model: {model}, Detector: {detector_used}")
        
        return jsonify({
            'match': match,
            'confidence': round(confidence, 2),
            'distance': round(distance, 4),
            'threshold': round(threshold, 4),
            'model': model,
            'detector': detector_used
        }), 200

    except ValueError as e:
        logger.error(f"[{request_id}] Face detection failed: {str(e)}", exc_info=True)
        return jsonify({
            'match': False,
            'confidence': 0.0,
            'error': 'Face not detected in one or both images'
        }), 200
    except KeyError as e:
        logger.error(f"[{request_id}] Missing required field: {str(e)}")
        return jsonify({'error': f'Missing required field: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"[{request_id}] Error processing request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary files (original and preprocessed)
        if img1_path and os.path.exists(img1_path):
            os.unlink(img1_path)
            logger.debug(f"[{request_id}] Deleted temporary file: {img1_path}")
            # Also delete preprocessed version
            preprocessed1 = img1_path.replace('.jpg', '_preprocessed.jpg').replace('.png', '_preprocessed.png')
            if os.path.exists(preprocessed1):
                os.unlink(preprocessed1)
                logger.debug(f"[{request_id}] Deleted preprocessed file: {preprocessed1}")
        
        if img2_path and os.path.exists(img2_path):
            os.unlink(img2_path)
            logger.debug(f"[{request_id}] Deleted temporary file: {img2_path}")
            # Also delete preprocessed version
            preprocessed2 = img2_path.replace('.jpg', '_preprocessed.jpg').replace('.png', '_preprocessed.png')
            if os.path.exists(preprocessed2):
                os.unlink(preprocessed2)
                logger.debug(f"[{request_id}] Deleted preprocessed file: {preprocessed2}")

if __name__ == '__main__':
    logger.info("Starting Face Verification Service on port 5001")
    app.run(port=5001, debug=True)