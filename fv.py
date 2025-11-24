import json
import os
import boto3
from dotenv import load_dotenv
from deepface import DeepFace
from urllib.parse import urlparse
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import logging
import threading
import queue

# Production-ready logging: Only WARNING and above for libraries
logging.basicConfig(
    level=logging.WARNING,  # Suppress INFO/DEBUG from libraries
    format='[FACE-EMBED] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Keep INFO for our app

# Silence verbose libraries
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('deepface').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('boto3').setLevel(logging.WARNING)

load_dotenv()

app = Flask(__name__)
CORS(app)

def run_with_timeout(func, args=(), kwargs=None, timeout=15):
    """Run function with timeout."""
    if kwargs is None:
        kwargs = {}
    q = queue.Queue()
    def wrapper():
        try:
            q.put((True, func(*args, **kwargs)))
        except Exception as e:
            q.put((False, e))
    t = threading.Thread(target=wrapper)
    t.daemon = True
    t.start()
    try:
        success, result = q.get(timeout=timeout)
        return result if success else None
    except queue.Empty:
        raise TimeoutError(f"Timeout after {timeout}s")

def detect_and_crop_face(image_path):
    """
    Detect face and return cropped, padded face region for better embedding extraction.
    This is the KEY fix for distant faces.
    """
    img = cv2.imread(image_path)
    if img is None:
        pil_img = Image.open(image_path)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    h, w = img.shape[:2]
    
    # Use OpenCV for fast initial detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        # Try with more lenient parameters
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
    
    if len(faces) == 0:
        logger.info("No face detected by OpenCV, using full image")
        return image_path, None, False
    
    # Get largest face
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    fx, fy, fw, fh = largest_face
    
    # Calculate face area ratio
    face_area_ratio = (fw * fh) / (w * h)
    is_distant = face_area_ratio < 0.08  # Less than 8% = distant face
    
    if not is_distant:
        # Face is large enough, return original
        return image_path, {
            'x': int(fx), 'y': int(fy), 'w': int(fw), 'h': int(fh),
            'area_ratio': face_area_ratio,
            'is_distant': False
        }, False
    
    logger.info(f"[DISTANT-FACE] Processing small face ({face_area_ratio*100:.1f}% of image)")
    
    # === DISTANT FACE HANDLING ===
    # Crop and upscale the face region for better embedding
    
    # Add padding around face (50% on each side)
    padding = 0.5
    pad_w = int(fw * padding)
    pad_h = int(fh * padding)
    
    x1 = max(0, fx - pad_w)
    y1 = max(0, fy - pad_h)
    x2 = min(w, fx + fw + pad_w)
    y2 = min(h, fy + fh + pad_h)
    
    # Crop face region
    face_crop = img[y1:y2, x1:x2]
    crop_h, crop_w = face_crop.shape[:2]
    
    # Upscale cropped face to at least 400x400 for better features
    target_size = 400
    min_dim = min(crop_w, crop_h)
    
    if min_dim < target_size:
        scale = target_size / min_dim
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        face_crop = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Apply enhancement to upscaled face
    # CLAHE for contrast
    lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    face_crop = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    # Light sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    face_crop = cv2.filter2D(face_crop, -1, kernel)
    
    # Save cropped face
    cropped_path = image_path.replace(os.path.splitext(image_path)[1], '_face_crop.jpg')
    cv2.imwrite(cropped_path, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return cropped_path, {
        'x': int(fx), 'y': int(fy), 'w': int(fw), 'h': int(fh),
        'area_ratio': face_area_ratio,
        'is_distant': True,
        'crop_applied': True,
        'original_size': f"{w}x{h}",
        'crop_size': f"{face_crop.shape[1]}x{face_crop.shape[0]}"
    }, True

def enhance_image_light(image_path):
    """Light enhancement - only when needed."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        
        h, w = img.shape[:2]
        
        # Upscale if small
        min_dim = min(h, w)
        if min_dim < 640:
            scale = 640 / min_dim
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LANCZOS4)
        
        # CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        enhanced_path = image_path.replace(os.path.splitext(image_path)[1], '_enhanced.jpg')
        cv2.imwrite(enhanced_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return enhanced_path
    except:
        return image_path

def extract_embedding(image_path, timeout_seconds=12):
    """Extract embedding with fallback strategies."""
    strategies = [
        ("Facenet512", "retinaface"),
        ("Facenet512", "ssd"),
        ("Facenet512", "opencv"),
        ("ArcFace", "opencv")
    ]
    
    for model, detector in strategies:
        try:
            result = run_with_timeout(
                DeepFace.represent,
                kwargs={
                    'img_path': image_path,
                    'model_name': model,
                    'enforce_detection': False,  # Don't fail on weak detection
                    'detector_backend': detector,
                    'align': True
                },
                timeout=timeout_seconds
            )
            
            if result and len(result) > 0:
                logger.info(f"[EMBEDDING] Extracted using {model}/{detector}")
                return result[0], model, detector
                
        except Exception as e:
            continue
    
    return None, None, None

def get_face_embedding(s3_url):
    """Main embedding extraction with distant face optimization."""
    bucket_name = os.getenv('BUCKET_NAME')
    bucket_region = os.getenv('BUCKET_REGION')
    access_key = os.getenv('ACCESS_KEY')
    secret_key = os.getenv('SECRET_KEY')
    
    if not all([bucket_name, bucket_region, access_key, secret_key]):
        raise ValueError("Missing S3 credentials")
    
    s3_client = boto3.client('s3', region_name=bucket_region,
                             aws_access_key_id=access_key,
                             aws_secret_access_key=secret_key)
    
    # Parse S3 URL
    if s3_url.startswith('s3://'):
        parsed = urlparse(s3_url)
        bucket = parsed.netloc or bucket_name
        key = parsed.path.lstrip('/')
    elif 'http' in s3_url:
        parsed = urlparse(s3_url)
        if '.s3.' in parsed.hostname:
            bucket = parsed.hostname.split('.s3.')[0]
            key = parsed.path.lstrip('/')
        else:
            bucket = bucket_name
            key = parsed.path.lstrip('/')
    else:
        raise ValueError(f"Invalid S3 URL: {s3_url}")
    
    file_ext = os.path.splitext(key)[1] or '.jpg'
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext).name
    temp_files = [temp_path]
    
    try:
        s3_client.download_file(bucket, key, temp_path)
        
        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            raise ValueError("Empty file")
        
        # === KEY OPTIMIZATION: Detect and handle distant faces ===
        processed_path, face_info, is_distant = detect_and_crop_face(temp_path)
        
        if processed_path != temp_path:
            temp_files.append(processed_path)
        
        # Extract embedding from processed image
        result, model_used, detector_used = extract_embedding(processed_path)
        
        # If failed, try with light enhancement
        if result is None:
            enhanced_path = enhance_image_light(temp_path)
            temp_files.append(enhanced_path)
            result, model_used, detector_used = extract_embedding(enhanced_path)
        
        if result is None:
            return {
                "success": False,
                "error": "Could not detect face",
                "details": "No face found after all attempts"
            }
        
        embedding = result["embedding"]
        facial_area = result.get("facial_area", face_info or {})
        
        # Calculate quality metrics
        if face_info:
            area_ratio = face_info.get('area_ratio', 0)
        else:
            # Estimate from DeepFace result
            fa = facial_area
            if fa and 'w' in fa and 'h' in fa:
                img = cv2.imread(temp_path)
                if img is not None:
                    h, w = img.shape[:2]
                    area_ratio = (fa['w'] * fa['h']) / (w * h)
                else:
                    area_ratio = 0.1
            else:
                area_ratio = 0.1
        
        # Determine confidence based on face size
        if area_ratio >= 0.15:
            face_confidence = "high"
        elif area_ratio >= 0.05:
            face_confidence = "medium"
        else:
            face_confidence = "low"
        
        logger.info(f"[SUCCESS] Confidence: {face_confidence}, Area: {area_ratio*100:.1f}%")
        
        return {
            "success": True,
            "embedding": embedding,
            "facial_area": facial_area,
            "model_used": model_used,
            "detector_used": detector_used,
            "face_confidence": face_confidence,
            "face_metrics": {
                "area_ratio": round(area_ratio, 4),
                "is_distant": is_distant,
                "crop_applied": is_distant,
                "quality_score": round(min(1.0, area_ratio * 10), 2)  # 0-1 score
            }
        }
        
    except Exception as e:
        logger.error(f"[ERROR] Extraction failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "details": "Face detection failed"
        }
    finally:
        for path in temp_files:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

@app.route('/api/face-embedding', methods=['POST'])
def extract_face_embedding():
    try:
        data = request.get_json()
        if not data or 's3_url' not in data:
            return jsonify({"success": False, "error": "Missing 's3_url'"}), 400
        
        s3_url = data['s3_url']
        
        result = get_face_embedding(s3_url)
        status_code = 200 if result.get('success') else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"[API-ERROR] {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "version": "3.0-distant-optimized"}), 200

if __name__ == "__main__":
    logger.info("[STARTUP] Face Embedding API v3.0 starting on port 5010")
    app.run(host='0.0.0.0', port=5010, debug=False)