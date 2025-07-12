import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Project Paths (Simplified for Local VS Code) ---
try:
    # This works when running as a script
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for interactive environments like a Jupyter cell in VS Code
    APP_ROOT = os.getcwd()

# ### UPDATED ### Direct path to the dataset folder
DATASET_DIR = os.path.join(APP_ROOT, 'dataset') 
MODELS_OUTPUT_DIR = os.path.join(APP_ROOT, 'models_output')

# --- Configuration ---
IMG_WIDTH, IMG_HEIGHT = 128, 128
SIFT_MAX_FEATURES = 64

def check_gpu():
    """Checks for available GPUs and configures them for TensorFlow."""
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

# ### REMOVED ### The unzip_dataset function is no longer needed for a local setup.

def load_and_preprocess_image_cnn(image_path):
    """Loads and preprocesses a single image for the CNN branch."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"CNN Preprocessing: Could not read image {image_path}. Skipping.")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        return img
    except Exception as e:
        logger.error(f"Error processing {image_path} for CNN: {e}", exc_info=True)
        return None

def extract_sift_features_for_model(image_path):
    """Extracts flattened SIFT features from a single image."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"SIFT Features: Could not read grayscale image {image_path}. Returning zeros.")
            return np.zeros(SIFT_MAX_FEATURES * 128)

        sift = cv2.SIFT_create(nfeatures=SIFT_MAX_FEATURES)
        _, descriptors = sift.detectAndCompute(img, None)

        if descriptors is None or len(descriptors) == 0:
            return np.zeros(SIFT_MAX_FEATURES * 128)

        # Pad or truncate descriptors to a fixed size
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
    """Loads all images and SIFT features from the genuine and forged folders."""
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
        raise ValueError("No images were loaded. Check dataset paths and image files.")
    
    # Pre-process images for MobileNetV2 after loading
    cnn_images_np = np.array(cnn_images)
    cnn_images_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(cnn_images_np)

    return cnn_images_preprocessed, np.array(sift_features_list), np.array(labels)

def calculate_class_weights(labels):
    """Calculates class weights for an imbalanced dataset."""
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return {i: weights[i] for i in range(len(weights))}

def build_advanced_model(sift_input_shape_dim):
    """Builds the advanced multi-input model using Transfer Learning."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    cnn_input_layer = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name='cnn_input')

    augmentation_layer = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.2),
    ], name='augmentation')

    x = augmentation_layer(cnn_input_layer)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
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

    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    final_model.summary(print_fn=logger.info)
    return final_model

def train_and_save_model():
    """Main function to orchestrate the training and saving process."""
    check_gpu()

    # ### UPDATED ### Define data paths directly and check for existence
    genuine_path = os.path.join(DATASET_DIR, 'genuine')
    forged_path = os.path.join(DATASET_DIR, 'forged')

    logger.info(f"Looking for genuine signatures in: {genuine_path}")
    logger.info(f"Looking for forged signatures in: {forged_path}")

    if not os.path.isdir(genuine_path) or not os.path.isdir(forged_path):
        error_msg = f"Dataset folders not found. Please ensure 'genuine' and 'forged' folders exist inside '{DATASET_DIR}'."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    cnn_images, sift_features, labels = load_data(genuine_path, forged_path)

    class_weights = calculate_class_weights(labels)
    logger.info(f"Calculated Class Weights: {class_weights}")

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
    model = build_advanced_model(sift_input_dim)

    callbacks_list = [
        EarlyStopping(monitor='val_recall', mode='max', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    ]

    logger.info("Starting model training with local dataset...")
    history = model.fit(
        [X_cnn_train, X_sift_train_scaled], y_train,
        validation_data=([X_cnn_test, X_sift_test_scaled], y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1
    )

    results = model.evaluate([X_cnn_test, X_sift_test_scaled], y_test, verbose=0)
    
    logger.info("\n--- Final Test Results ---")
    try:
        recall_index = model.metrics_names.index('recall')
        recall_value = results[recall_index]
    except ValueError:
        recall_value = "N/A"
        
    for name, value in zip(model.metrics_names, results):
        logger.info(f"{name.capitalize()}: {value:.4f}")
    
    logger.info("\n--- Summary ---")
    if recall_value != "N/A":
        logger.info(f"Target Metric (Recall): {recall_value*100:.2f}%")
        logger.info(f"This means the model correctly identified {recall_value*100:.2f}% of all forged signatures in the test set.")
    logger.info("------------------------")

    os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
    model_save_path = os.path.join(MODELS_OUTPUT_DIR, 'signature_advanced_model.keras')
    scaler_save_path = os.path.join(MODELS_OUTPUT_DIR, 'sift_feature_scaler_advanced.pkl')

    model.save(model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(sift_scaler, f)
    logger.info(f"SIFT scaler saved to {scaler_save_path}")
    logger.info("Training complete.")

if __name__ == '__main__':
    train_and_save_model()