import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import zipfile
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Project Root and Paths ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
ZIP_FILE_PATH = os.path.join(APP_ROOT, 'dataset.zip')
EXTRACTION_PATH = os.path.join(APP_ROOT, 'extracted_dataset_local')
MODELS_OUTPUT_DIR = os.path.join(APP_ROOT, 'models_output')

# --- Configuration (same as your Colab script) ---
IMG_WIDTH, IMG_HEIGHT = 128, 128
SIFT_MAX_FEATURES = 64

def check_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPUs available: {len(gpus)}")
        except RuntimeError as e:
            logger.error(f"Error setting up GPU: {e}")
    else:
        logger.info("No GPUs available, using CPU.")

def unzip_dataset():
    if not os.path.exists(ZIP_FILE_PATH):
        logger.error(f"Dataset zip file not found at: {ZIP_FILE_PATH}")
        raise FileNotFoundError(f"Dataset zip file not found at: {ZIP_FILE_PATH}")

    if os.path.exists(EXTRACTION_PATH):
        logger.info(f"Removing existing extraction directory: {EXTRACTION_PATH}")
        shutil.rmtree(EXTRACTION_PATH)
    os.makedirs(EXTRACTION_PATH, exist_ok=True)

    logger.info(f"Attempting to unzip {ZIP_FILE_PATH} to {EXTRACTION_PATH}...")
    try:
        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACTION_PATH)
        logger.info(f"Successfully unzipped dataset to {EXTRACTION_PATH}")

        extracted_items = os.listdir(EXTRACTION_PATH)
        if len(extracted_items) == 1 and os.path.isdir(os.path.join(EXTRACTION_PATH, extracted_items[0])):
            dataset_root = os.path.join(EXTRACTION_PATH, extracted_items[0])
        else:
            dataset_root = EXTRACTION_PATH
        
        genuine_p = os.path.join(dataset_root, 'genuine')
        forged_p = os.path.join(dataset_root, 'forged')

        if not os.path.exists(genuine_p) or not os.path.exists(forged_p):
            logger.error(f"Error: 'genuine' ({genuine_p}) or 'forged' ({forged_p}) not found post-unzip.")
            logger.error(f"Contents of {EXTRACTION_PATH}: {os.listdir(EXTRACTION_PATH)}")
            if os.path.exists(dataset_root):
                 logger.error(f"Contents of {dataset_root}: {os.listdir(dataset_root)}")
            raise FileNotFoundError("Could not locate 'genuine' and 'forged' subdirectories.")
        return genuine_p, forged_p
    except Exception as e:
        logger.error(f"An error occurred during unzipping or path setup: {e}", exc_info=True)
        raise

def load_and_preprocess_image_cnn(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"CNN Preprocessing: Could not read image {image_path}. Skipping.")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype('float32') / 255.0
        return img
    except Exception as e:
        logger.error(f"Error processing {image_path} for CNN: {e}", exc_info=True)
        return None

def extract_sift_features_for_model(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"SIFT Features: Could not read grayscale image {image_path}. Returning zeros.")
            return np.zeros(SIFT_MAX_FEATURES * 128)

        sift = cv2.SIFT_create(nfeatures=SIFT_MAX_FEATURES)
        _, descriptors = sift.detectAndCompute(img, None)

        if descriptors is None or len(descriptors) == 0:
            return np.zeros(SIFT_MAX_FEATURES * 128)

        if len(descriptors) < SIFT_MAX_FEATURES:
            padding = np.zeros((SIFT_MAX_FEATURES - len(descriptors), 128))
            descriptors = np.vstack((descriptors, padding))
        elif len(descriptors) > SIFT_MAX_FEATURES:
            descriptors = descriptors[:SIFT_MAX_FEATURES]
        return descriptors.flatten()
    except Exception as e:
        logger.error(f"Error extracting SIFT from {image_path}: {e}", exc_info=True)
        return np.zeros(SIFT_MAX_FEATURES * 128)

def load_data(genuine_path, forged_path):
    cnn_images, sift_features_list, labels = [], [], []
    logger.info("Loading genuine signatures...")
    for filename in os.listdir(genuine_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            img_path = os.path.join(genuine_path, filename)
            cnn_img = load_and_preprocess_image_cnn(img_path)
            sift_feat = extract_sift_features_for_model(img_path)
            if cnn_img is not None:
                cnn_images.append(cnn_img)
                sift_features_list.append(sift_feat)
                labels.append(0) # 0 for genuine
    logger.info(f"Loaded {len(cnn_images)} genuine signatures.")
    genuine_count = len(cnn_images)

    logger.info("Loading forged signatures...")
    for filename in os.listdir(forged_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            img_path = os.path.join(forged_path, filename)
            cnn_img = load_and_preprocess_image_cnn(img_path)
            sift_feat = extract_sift_features_for_model(img_path)
            if cnn_img is not None:
                cnn_images.append(cnn_img)
                sift_features_list.append(sift_feat)
                labels.append(1) # 1 for forged
    logger.info(f"Loaded {len(cnn_images) - genuine_count} forged signatures.")
    logger.info(f"Total samples: {len(cnn_images)}")

    if not cnn_images:
        raise ValueError("No images were loaded. Check dataset paths and image files after unzipping.")
    return np.array(cnn_images), np.array(sift_features_list), np.array(labels)

def build_model(sift_input_shape_dim):
    cnn_input_layer = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name='cnn_input')
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(cnn_input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    cnn_output_branch = Dense(128, activation='relu')(x)

    sift_input_layer = Input(shape=(sift_input_shape_dim,), name='sift_input')
    y = Dense(128, activation='relu')(sift_input_layer)
    y = Dropout(0.3)(y)
    sift_output_branch = Dense(64, activation='relu')(y)

    combined_branches = concatenate([cnn_output_branch, sift_output_branch])
    z = Dense(128, activation='relu')(combined_branches)
    z = Dropout(0.5)(z)
    output_layer = Dense(1, activation='sigmoid', name='output')(z)

    final_model = Model(inputs=[cnn_input_layer, sift_input_layer], outputs=output_layer)
    final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    final_model.summary(print_fn=logger.info)
    return final_model

def train_and_save_model():
    check_gpu()
    genuine_path, forged_path = unzip_dataset()
    cnn_images, sift_features, labels = load_data(genuine_path, forged_path)

    X_cnn_train, X_cnn_test, \
    X_sift_train, X_sift_test, \
    y_train, y_test = train_test_split(
        cnn_images, sift_features, labels,
        test_size=0.2, random_state=42, stratify=labels
    )
    logger.info(f"Training samples: {len(X_cnn_train)}, Testing samples: {len(X_cnn_test)}")

    sift_scaler = StandardScaler()
    X_sift_train_scaled = sift_scaler.fit_transform(X_sift_train)
    X_sift_test_scaled = sift_scaler.transform(X_sift_test)

    sift_input_dim = X_sift_train_scaled.shape[1]
    model = build_model(sift_input_dim)

    callbacks_list = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    ]

    logger.info("Starting model training...")
    history = model.fit(
        [X_cnn_train, X_sift_train_scaled], y_train,
        validation_data=([X_cnn_test, X_sift_test_scaled], y_test),
        epochs=50, # Adjust as needed
        batch_size=32,
        callbacks=callbacks_list,
        verbose=1
    )

    loss, accuracy = model.evaluate([X_cnn_test, X_sift_test_scaled], y_test, verbose=0)
    logger.info(f"\nTest Accuracy: {accuracy*100:.2f}%")
    logger.info(f"Test Loss: {loss:.4f}")

    os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
    model_save_path = os.path.join(MODELS_OUTPUT_DIR, 'signature_combined_model.h5') # Or .keras
    scaler_save_path = os.path.join(MODELS_OUTPUT_DIR, 'sift_feature_scaler.pkl')

    model.save(model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(sift_scaler, f)
    logger.info(f"SIFT scaler saved to {scaler_save_path}")
    logger.info("Training complete. Model and scaler saved.")

if __name__ == '__main__':
    train_and_save_model()