import os
import cv2
import gc
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# =========================
# Configuration
# =========================
CONFIG = {
    'PATCH_SIZE': 112,
    'STRIDE': 70,
    'MIN_INK': 0.02,
    'EPOCHS': 30,
    'BATCH_SIZE': 64,
    'VAL_SPLIT': 0.2
}

TRAIN_DIR = 'train'
MODEL_SAVE_PATH = 'model.keras'


# =========================
# Data Preparation
# =========================
def get_patches_and_labels(directory):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return np.array([]), np.array([])

    patches, labels = [], []

    files = sorted(
        [f for f in os.listdir(directory)
         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    )

    print(f"Processing {len(files)} training images...")

    for f in files:
        try:
            wid = int(f.split('_')[0])  # writer / class id

            img_path = os.path.join(directory, f)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            h, w = img.shape
            for y in range(0, h - CONFIG['PATCH_SIZE'], CONFIG['STRIDE']):
                for x in range(0, w - CONFIG['PATCH_SIZE'], CONFIG['STRIDE']):
                    crop = img[y:y+CONFIG['PATCH_SIZE'], x:x+CONFIG['PATCH_SIZE']]

                    ink_ratio = np.sum(crop < 220) / (CONFIG['PATCH_SIZE'] ** 2)
                    if ink_ratio > CONFIG['MIN_INK']:
                        patches.append(crop)
                        labels.append(wid)

        except Exception as e:
            print(f"Skipping file {f}: {e}")

    X = np.array(patches, dtype='float32') / 255.0
    X = np.expand_dims(X, axis=-1)
    y = np.array(labels, dtype='int32')

    return X, y


# =========================
# Model Architecture
# =========================
def build_model(num_classes):
    inputs = layers.Input(shape=(CONFIG['PATCH_SIZE'], CONFIG['PATCH_SIZE'], 1))

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)


# =========================
# Training & Evaluation
# =========================
if __name__ == "__main__":

    print("Loading data...")
    X, y = get_patches_and_labels(TRAIN_DIR)

    if len(X) == 0:
        print("No data loaded. Check training directory.")
        exit()

    # Encode labels
    unique_ids = np.unique(y)
    label_map = {uid: i for i, uid in enumerate(unique_ids)}
    y_encoded = np.array([label_map[uid] for uid in y])

    np.save('classes.npy', unique_ids)
    print(f"Classes saved ({len(unique_ids)} classes).")

    # Train / Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded,
        test_size=CONFIG['VAL_SPLIT'],
        stratify=y_encoded,
        random_state=42
    )

    # Build & compile model
    model = build_model(len(unique_ids))
    # Use accuracy for sparse multiclass labels. The AUC metric expects
    # matching label shapes (one-hot / probabilities) and can cause shape
    # mismatches with sparse integer labels.
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nStarting training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['EPOCHS'],
        batch_size=CONFIG['BATCH_SIZE'],
        verbose=1
    )

    # =========================
    # Explicit Evaluation 
    # =========================
    print("\nEvaluating model on validation set...")

    y_val_probs = model.predict(X_val)
    y_val_preds = np.argmax(y_val_probs, axis=1)

    # Accuracy
    val_accuracy = np.mean(y_val_preds == y_val)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_val_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_preds))

    # AUC (One-vs-Rest for multiclass)
    y_val_onehot = tf.keras.utils.to_categorical(
        y_val, num_classes=len(unique_ids)
    )
    auc_score = roc_auc_score(
        y_val_onehot, y_val_probs, multi_class='ovr'
    )
    print(f"Validation AUC (OvR): {auc_score:.4f}")

    # Save model
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")

    # Cleanup
    gc.collect()
