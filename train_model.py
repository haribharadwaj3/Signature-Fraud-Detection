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
# ### NEW ### - Import L2 regularizer
from tensorflow.keras.regularizers import l2
import logging
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Project Paths ---
try:
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    APP_ROOT = os.getcwd()

DATASET_DIR = os.path.join(APP_ROOT, 'dataset')
MODELS_OUTPUT_DIR = os.path.join(APP_ROOT, 'models_output')

# --- Configuration ---
IMG_WIDTH, IMG_HEIGHT = 128, 128
SIFT_MAX_FEATURES = 64
# ### MODIFIED ### - Adjusted epochs and added regularization factor
MAX_INITIAL_EPOCHS = 30  # Let EarlyStopping decide the actual number
MAX_FINE_TUNE_EPOCHS = 20 # Let EarlyStopping decide
L2_FACTOR = 1e-5         # L2 regularization strength

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

def load_and_preprocess_image_cnn(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None: return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        return img
    except Exception as e:
        logger.error(f"Error processing {image_path} for CNN: {e}", exc_info=True)
        return None

def extract_sift_features_for_model(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return np.zeros(SIFT_MAX_FEATURES * 128)
        sift = cv2.SIFT_create(nfeatures=SIFT_MAX_FEATURES)
        _, descriptors = sift.detectAndCompute(img, None)
        if descriptors is None or len(descriptors) == 0: return np.zeros(SIFT_MAX_FEATURES * 128)
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
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(genuine_path, filename)
            cnn_img, sift_feat = load_and_preprocess_image_cnn(img_path), extract_sift_features_for_model(img_path)
            if cnn_img is not None:
                cnn_images.append(cnn_img); sift_features_list.append(sift_feat); labels.append(0)
    genuine_count = len(cnn_images)
    logger.info(f"Loaded {genuine_count} genuine signatures.")
    logger.info("Loading forged signatures...")
    for filename in os.listdir(forged_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(forged_path, filename)
            cnn_img, sift_feat = load_and_preprocess_image_cnn(img_path), extract_sift_features_for_model(img_path)
            if cnn_img is not None:
                cnn_images.append(cnn_img); sift_features_list.append(sift_feat); labels.append(1)
    logger.info(f"Loaded {len(cnn_images) - genuine_count} forged signatures.")
    logger.info(f"Total samples: {len(cnn_images)}")
    if not cnn_images: raise ValueError("No images were loaded.")
    cnn_images_np = np.array(cnn_images)
    cnn_images_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(cnn_images_np)
    return cnn_images_preprocessed, np.array(sift_features_list), np.array(labels)

def calculate_class_weights(labels):
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    return {i: weights[i] for i in range(len(weights))}

# ### MODIFIED ### - Added L2 regularization to the Dense layers
def build_model(sift_input_shape_dim):
    """Builds the multi-input model with L2 regularization."""
    base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    cnn_input_layer = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name='cnn_input')
    augmentation_layer = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name='augmentation')
    
    x = augmentation_layer(cnn_input_layer)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x) # Strong dropout after GAP layer
    cnn_output_branch = Dense(128, activation='relu', kernel_regularizer=l2(L2_FACTOR))(x)

    sift_input_layer = Input(shape=(sift_input_shape_dim,), name='sift_input')
    y = Dense(128, activation='relu', kernel_regularizer=l2(L2_FACTOR))(sift_input_layer)
    y = Dropout(0.5)(y)
    sift_output_branch = Dense(64, activation='relu', kernel_regularizer=l2(L2_FACTOR))(y)

    combined_branches = concatenate([cnn_output_branch, sift_output_branch])
    z = Dense(128, activation='relu', kernel_regularizer=l2(L2_FACTOR))(combined_branches)
    z = Dropout(0.5)(z)
    output_layer = Dense(1, activation='sigmoid', name='output')(z)

    final_model = Model(inputs=[cnn_input_layer, sift_input_layer], outputs=output_layer)
    return final_model, base_model

def generate_and_log_report(model, X_test, y_test, title="FINAL PERFORMANCE REPORT"):
    logger.info("\n\n" + "="*50)
    logger.info(f"{title} ON TEST SET")
    logger.info("="*50)
    y_pred_prob = model.predict(X_test)
    y_pred_class = (y_pred_prob > 0.5).astype("int32").flatten()
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel()
    total = len(y_test); correct = tp + tn; wrong = fp + fn
    accuracy = correct / total; precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0; f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    report = f"""
| Category                      | Description                                        | Count / Value |
|:------------------------------|:---------------------------------------------------|:--------------|
| **Total Test Signatures**     | Total signatures in the test set                   | {total:<13} |
| **Correct Classifications**   | Signatures correctly identified (TP + TN)          | {correct:<13} |
| **Incorrect Classifications** | Signatures incorrectly identified (FP + FN)        | {wrong:<13} |
| --- Performance Breakdown --- |                                                    |               |
| **True Positives (TP)**       | Forged signature correctly identified as **Fraud**   | {tp:<13} |
| **True Negatives (TN)**       | Genuine signature correctly identified as **Genuine**| {tn:<13} |
| **False Positives (FP)**      | Genuine signature incorrectly identified as **Fraud**| {fp:<13} |
| **False Negatives (FN)**      | Forged signature incorrectly identified as **Genuine** | {fn:<13} |
| --- Final Evaluation Metrics ---| Formula                                          | Score         |
| **Accuracy**                  | (TP + TN) / Total                                  | {accuracy:.4f}        |
| **Precision (for Fraud)**     | TP / (TP + FP)                                     | {precision:.4f}       |
| **Recall (for Fraud)**        | TP / (TP + FN)                                     | {recall:.4f}        |
| **F1-Score (for Fraud)**      | 2 * (Prec * Rec) / (Prec + Rec)                    | {f1:.4f}        |
"""
    for line in report.split('\n'): logger.info(line)
    logger.info("="*50 + "\n")

def plot_training_history(history, fine_tune_history, save_dir):
    logger.info("Generating and saving combined training history plots...")
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if fine_tune_history is None
    if fine_tune_history:
        initial_epochs = len(history.history['accuracy'])
        acc = history.history['accuracy'] + fine_tune_history.history['accuracy']
        val_acc = history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
        loss = history.history['loss'] + fine_tune_history.history['loss']
        val_loss = history.history['val_loss'] + fine_tune_history.history['val_loss']
    else:
        initial_epochs = 0
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    if fine_tune_history:
        plt.axvline(x=initial_epochs-1, color='r', linestyle='--', label='Start Fine-Tuning')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    if fine_tune_history:
        plt.axvline(x=initial_epochs-1, color='r', linestyle='--', label='Start Fine-Tuning')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_history_plots_regularized.png')
    plt.savefig(plot_path); plt.close()
    logger.info(f"Training history plots saved to {plot_path}")

def train_and_save_model():
    check_gpu()
    genuine_path = os.path.join(DATASET_DIR, 'genuine')
    forged_path = os.path.join(DATASET_DIR, 'forged')
    if not os.path.isdir(genuine_path) or not os.path.isdir(forged_path):
        raise FileNotFoundError("Dataset folders not found.")

    cnn_images, sift_features, labels = load_data(genuine_path, forged_path)
    class_weights = calculate_class_weights(labels)
    logger.info(f"Calculated Class Weights: {class_weights}")

    X_cnn_train, X_cnn_test, X_sift_train, X_sift_test, y_train, y_test = train_test_split(
        cnn_images, sift_features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    sift_scaler = StandardScaler()
    X_sift_train_scaled = sift_scaler.fit_transform(X_sift_train)
    X_sift_test_scaled = sift_scaler.transform(X_sift_test)
    
    model, base_model = build_model(X_sift_train_scaled.shape[1])
    
    # ### MODIFIED ### - Refined EarlyStopping callback to monitor val_loss
    # We use two different callbacks for each phase.
    initial_callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', patience=7, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
    ]

    # ### PHASE 1: INITIAL TRAINING (FEATURE EXTRACTION) ###
    logger.info("\n--- PHASE 1: Training top layers with strong regularization ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )
    history = model.fit(
        [X_cnn_train, X_sift_train_scaled], y_train,
        validation_data=([X_cnn_test, X_sift_test_scaled], y_test),
        epochs=MAX_INITIAL_EPOCHS,
        batch_size=32,
        class_weight=class_weights,
        callbacks=initial_callbacks
    )
    
    # ### PHASE 2: FINE-TUNING ###
    logger.info("\n--- PHASE 2: Unfreezing and Fine-Tuning ---")
    base_model.trainable = True
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Use a very low learning rate for fine-tuning
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )
    
    fine_tune_callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1)
    ]
    
    logger.info("Continuing training with fine-tuning...")
    # The number of epochs here is a max; EarlyStopping will likely stop it sooner.
    total_epochs = len(history.epoch) + MAX_FINE_TUNE_EPOCHS
    fine_tune_history = model.fit(
        [X_cnn_train, X_sift_train_scaled], y_train,
        validation_data=([X_cnn_test, X_sift_test_scaled], y_test),
        epochs=total_epochs,
        initial_epoch=len(history.epoch),
        batch_size=32,
        class_weight=class_weights,
        callbacks=fine_tune_callbacks
    )

    # --- FINAL EVALUATION AND SAVING ---
    generate_and_log_report(model, [X_cnn_test, X_sift_test_scaled], y_test, title="FINAL PERFORMANCE REPORT (Regularized & Fine-Tuned)")
    plot_training_history(history, fine_tune_history, MODELS_OUTPUT_DIR)

    os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
    model.save(os.path.join(MODELS_OUTPUT_DIR, 'signature_model_final.keras'))
    with open(os.path.join(MODELS_OUTPUT_DIR, 'sift_scaler_final.pkl'), 'wb') as f:
        pickle.dump(sift_scaler, f)
    logger.info("Final regularized model and scaler saved successfully.")

if __name__ == '__main__':
    train_and_save_model()