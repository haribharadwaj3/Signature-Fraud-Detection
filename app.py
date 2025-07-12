import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from flask import Flask, request, render_template, url_for, redirect 
from werkzeug.utils import secure_filename
import shutil
import logging
import time
import base64 # <-- Added for Base64 encoding

# --- Project Root and Paths ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER_TEMP = os.path.join(APP_ROOT, 'uploads_temp')
STATIC_FOLDER = os.path.join(APP_ROOT, 'static')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}
MODELS_INPUT_DIR = os.path.join(APP_ROOT, 'models_output')
MODEL_FILENAME = 'signature_advanced_model.keras'
SIFT_SCALER_FILENAME = 'sift_feature_scaler_advanced.pkl'
MODEL_PATH_FLASK = os.path.join(MODELS_INPUT_DIR, MODEL_FILENAME)
SIFT_SCALER_PATH_FLASK = os.path.join(MODELS_INPUT_DIR, SIFT_SCALER_FILENAME)
IMG_WIDTH, IMG_HEIGHT = 128, 128
SIFT_MAX_FEATURES = 64
SIFT_DIRECT_RATIO_THRESH = 0.70
MIN_SIFT_DIRECT_MATCHES = 10

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = "flask_app_secret_key_for_signatures_v2"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
flask_logger = logging.getLogger(__name__)
os.makedirs(UPLOAD_FOLDER_TEMP, exist_ok=True)

# --- Load Model and Scaler ---
model_loaded = None
sift_scaler_loaded = None
try:
    if not os.path.exists(MODEL_PATH_FLASK):
        flask_logger.error(f"Model file not found at: {MODEL_PATH_FLASK}. Please run train_model.py first.")
    elif not os.path.exists(SIFT_SCALER_PATH_FLASK):
        flask_logger.error(f"SIFT scaler file not found at: {SIFT_SCALER_PATH_FLASK}. Please run train_model.py first.")
    else:
        flask_logger.info(f"Loading model from: {MODEL_PATH_FLASK}")
        model_loaded = tf.keras.models.load_model(MODEL_PATH_FLASK)
        flask_logger.info("Model loaded successfully for Flask app.")
        flask_logger.info(f"Loading SIFT scaler from: {SIFT_SCALER_PATH_FLASK}")
        with open(SIFT_SCALER_PATH_FLASK, 'rb') as f:
            sift_scaler_loaded = pickle.load(f)
        flask_logger.info("SIFT scaler loaded successfully for Flask app.")
except Exception as e:
    flask_logger.error(f"Error loading model or scaler for Flask app: {e}", exc_info=True)

# --- Helper Functions ---
def allowed_file_flask(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_cnn_flask(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None: return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
        img_normalized = img_resized.astype('float32') / 255.0
        return np.expand_dims(img_normalized, axis=0)
    except Exception as e:
        flask_logger.error(f"Error in preprocess_image_cnn_flask for {image_path}: {e}", exc_info=True)
        return None

def extract_sift_features_flask(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return np.zeros((1, SIFT_MAX_FEATURES * 128))
        sift = cv2.SIFT_create(nfeatures=SIFT_MAX_FEATURES)
        _, descriptors = sift.detectAndCompute(img, None)
        if descriptors is None or len(descriptors) == 0:
            return np.zeros((1, SIFT_MAX_FEATURES * 128))
        if len(descriptors) < SIFT_MAX_FEATURES:
            padding = np.zeros((SIFT_MAX_FEATURES - len(descriptors), 128))
            descriptors = np.vstack((descriptors, padding))
        elif len(descriptors) > SIFT_MAX_FEATURES:
            descriptors = descriptors[:SIFT_MAX_FEATURES]
        return descriptors.flatten().reshape(1, -1)
    except Exception as e:
        flask_logger.error(f"Error in extract_sift_features_flask for {image_path}: {e}", exc_info=True)
        return np.zeros((1, SIFT_MAX_FEATURES * 128))

def compare_sift_direct_flask(img_path1, img_path2):
    try:
        img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None: return False, 0
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2: return False, 0
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < SIFT_DIRECT_RATIO_THRESH * n.distance:
                    good_matches.append(m)
        return len(good_matches) >= MIN_SIFT_DIRECT_MATCHES, len(good_matches)
    except Exception as e:
        flask_logger.error(f"Error in compare_sift_direct_flask: {e}", exc_info=True)
        return False, 0

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index_page():
    if model_loaded is None or sift_scaler_loaded is None:
        page_error_msg = "Model or SIFT scaler not loaded. Please ensure training (run train_model.py) was successful and check application logs."
        return render_template('index.html', error_message_page=page_error_msg)

    result_text = None
    result_category_class = None
    request_error_msg = None
    img1_path_temp_abs = None
    img2_path_temp_abs = None
    
    # Variables for re-displaying images
    image1_data_url = None
    image1_filename = None
    image2_data_url = None
    image2_filename = None

    if request.method == 'POST':
        if 'image1' not in request.files or 'image2' not in request.files:
            request_error_msg = "Both image files must be uploaded."
        else:
            file1 = request.files['image1']
            file2 = request.files['image2']

            if file1.filename == '' or file2.filename == '':
                request_error_msg = "Please select two image files."
            elif not (allowed_file_flask(file1.filename) and allowed_file_flask(file2.filename)):
                request_error_msg = f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            else:
                ts = str(int(time.time()))
                base1, ext1 = os.path.splitext(secure_filename(file1.filename))
                base2, ext2 = os.path.splitext(secure_filename(file2.filename))
                unique_fn1 = f"{base1}_{ts}_1{ext1}"
                unique_fn2 = f"{base2}_{ts}_2{ext2}"

                img1_path_temp_abs = os.path.join(UPLOAD_FOLDER_TEMP, unique_fn1)
                img2_path_temp_abs = os.path.join(UPLOAD_FOLDER_TEMP, unique_fn2)
                try:
                    file1.save(img1_path_temp_abs)
                    file2.save(img2_path_temp_abs)

                    # --- Prepare images for re-display ---
                    try:
                        # Store original filenames from the request for display
                        image1_filename = file1.filename
                        image2_filename = file2.filename

                        with open(img1_path_temp_abs, "rb") as f:
                            img1_bytes_for_display = f.read()
                        img1_b64_for_display = base64.b64encode(img1_bytes_for_display).decode('utf-8')
                        image1_data_url = f"data:{file1.mimetype};base64,{img1_b64_for_display}"

                        with open(img2_path_temp_abs, "rb") as f:
                            img2_bytes_for_display = f.read()
                        img2_b64_for_display = base64.b64encode(img2_bytes_for_display).decode('utf-8')
                        image2_data_url = f"data:{file2.mimetype};base64,{img2_b64_for_display}"
                    
                    except Exception as e:
                        flask_logger.error(f"Error preparing images for re-display: {e}", exc_info=True)
                        # Images might not display, but processing can continue if files were saved
                
                except Exception as e:
                    flask_logger.error(f"Error saving uploaded files: {e}", exc_info=True)
                    request_error_msg = "Server error saving uploaded files."
                
                if not request_error_msg: # Proceed if files saved okay
                    cnn_input_img2 = preprocess_image_cnn_flask(img2_path_temp_abs)
                    sift_input_img2_raw = extract_sift_features_flask(img2_path_temp_abs)

                    if cnn_input_img2 is None or sift_input_img2_raw is None:
                        request_error_msg = "Error processing uploaded Image 2 for model input."
                    else:
                        try:
                            sift_input_img2_scaled = sift_scaler_loaded.transform(sift_input_img2_raw)
                            prediction_prob_img2 = model_loaded.predict([cnn_input_img2, sift_input_img2_scaled])[0][0]
                            is_img2_forged_by_model = prediction_prob_img2 > 0.5
                            sift_direct_match_flag, num_sift_matches = compare_sift_direct_flask(img1_path_temp_abs, img2_path_temp_abs)

                            if is_img2_forged_by_model:
                                result_category_class = "forged"
                                result_text = (f"Test signature (Image 2) is classified as FORGED by the model.\n"
                                               f"(Model's probability of forgery: {prediction_prob_img2:.2%})")
                            else:
                                if sift_direct_match_flag:
                                    result_category_class = "genuine"
                                    result_text = (f"Test signature (Image 2) is GENUINE (by model) AND "
                                                   f"it visually matches the reference (Image 1).\n"
                                                   f"(Model P(forgery): {prediction_prob_img2:.2%})\n"
                                                   f"(SIFT direct matches with reference: {num_sift_matches})")
                                else:
                                    result_category_class = "mismatch"
                                    result_text = (f"Test signature (Image 2) is GENUINE (by model), BUT "
                                                   f"it does NOT sufficiently match the reference (Image 1).\n"
                                                   f"(Model P(forgery): {prediction_prob_img2:.2%})\n"
                                                   f"(SIFT direct matches: {num_sift_matches}, required: {MIN_SIFT_DIRECT_MATCHES})")
                        except Exception as e:
                            flask_logger.error(f"Error during prediction or SIFT comparison: {e}", exc_info=True)
                            request_error_msg = "Error during signature analysis."
        
        # Cleanup temporary files
        if img1_path_temp_abs and os.path.exists(img1_path_temp_abs):
            try:
                os.remove(img1_path_temp_abs)
            except OSError as e:
                flask_logger.warning(f"Error removing temp file {img1_path_temp_abs}: {e}")
        if img2_path_temp_abs and os.path.exists(img2_path_temp_abs):
            try:
                os.remove(img2_path_temp_abs)
            except OSError as e:
                flask_logger.warning(f"Error removing temp file {img2_path_temp_abs}: {e}")
        
        # Render with all accumulated data, including image data for re-display
        return render_template('index.html', 
                               error_message_page=request_error_msg,
                               result_message_text=result_text,
                               result_class=result_category_class,
                               image1_data_url=image1_data_url,
                               image1_filename=image1_filename,
                               image2_data_url=image2_data_url,
                               image2_filename=image2_filename)

    # For GET request
    return render_template('index.html',
                           error_message_page=None, # No error for initial GET
                           result_message_text=None,
                           result_class=None,
                           image1_data_url=None, # No images on initial GET
                           image1_filename=None,
                           image2_data_url=None,
                           image2_filename=None)

if __name__ == '__main__':
    flask_logger.info(f"Flask app starting. TensorFlow version: {tf.__version__}")
    flask_logger.info(f"To run training, execute: python train_model.py")
    flask_logger.info(f"Ensure model files exist in '{MODELS_INPUT_DIR}' before starting if not training now.")
    app.run(debug=True, host='0.0.0.0', port=5000)