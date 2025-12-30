"""
run.py
------
Purpose:
- Load trained CNN model
- Perform inference on unseen test images
- Output predicted labels to CSV

NOTE:
- No training or model evaluation is performed here.
- All training and evaluation (Accuracy, AUC, Confusion Matrix)
  are done in train.py.
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

TEST_DIR = 'test'
MODEL_PATH = 'model.keras'
CLASSES_PATH = 'classes.npy'
OUTPUT_FILE = 'result.csv'
PATCH_SIZE = 112


def load_resources():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        print("Error: model.keras or classes.npy not found.")
        return None, None

    print("Loading model (CPU mode)...")

    use_quantize_scope = False
    try:
        from tensorflow_model_optimization.quantization.keras import quantize_scope
        use_quantize_scope = True
    except Exception:
        pass

    if use_quantize_scope:
        try:
            with quantize_scope():
                model = tf.keras.models.load_model(MODEL_PATH)
        except Exception:
            use_quantize_scope = False

    if not use_quantize_scope:
        try:
            from tensorflow.keras.layers import Dense as KerasDense
        except Exception:
            from keras.layers import Dense as KerasDense

        class DenseNoQuant(KerasDense):
            def __init__(self, *args, **kwargs):
                kwargs.pop('quantization_config', None)
                super().__init__(*args, **kwargs)

        custom_objects = {
            'Dense': DenseNoQuant,
            'tensorflow.keras.layers.Dense': DenseNoQuant,
        }

        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects=custom_objects
        )

    classes = np.load(CLASSES_PATH)
    return model, classes


def extract_patches(img):
    patches = []
    h, w = img.shape

    if h < PATCH_SIZE or w < PATCH_SIZE:
        pad_h = max(0, PATCH_SIZE - h)
        pad_w = max(0, PATCH_SIZE - w)
        img = cv2.copyMakeBorder(
            img, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=255
        )
        h, w = img.shape

    stride = 60

    for y in range(0, h - PATCH_SIZE + 1, stride):
        for x in range(0, w - PATCH_SIZE + 1, stride):
            crop = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            if np.sum(crop < 220) / (PATCH_SIZE**2) > 0.02:
                patches.append(crop)

    return np.array(patches, dtype='float32') / 255.0


def main():
    model, unique_ids = load_resources()
    if model is None:
        return

    inv_label_map = {i: uid for i, uid in enumerate(unique_ids)}

    if not os.path.exists(TEST_DIR):
        print(f"Error: Test folder '{TEST_DIR}' not found.")
        return

    test_files = sorted(
        [f for f in os.listdir(TEST_DIR)
         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    )

    print(f"Running inference on {len(test_files)} test images...")

    results = []
    correct_count = 0
    total_count = 0

    for f in test_files:
        try:
            img_path = os.path.join(TEST_DIR, f)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            patches = extract_patches(img)

            # try to infer true label from filename prefix (e.g. "12_...png")
            actual_label = None
            try:
                actual_label = int(f.split('_')[0])
            except Exception:
                actual_label = None

            if len(patches) == 0:
                pred_label = unique_ids[0]
                confidence = 0.0
            else:
                patches = np.expand_dims(patches, axis=-1)
                probs = model.predict(patches, verbose=0)
                summed_probs = np.sum(probs, axis=0)
                total = np.sum(summed_probs)
                pred_idx = np.argmax(summed_probs)
                pred_label = inv_label_map[pred_idx]
                confidence = float(summed_probs[pred_idx] / total) if total > 0 else 0.0

            results.append({
                'filename': f,
                'predicted_label': pred_label,
                'predicted_confidence': confidence,
                'actual_label': actual_label
            })

            # update overall counters when actual label known
            if actual_label is not None:
                total_count += 1
                if pred_label == actual_label:
                    correct_count += 1

            print(f"{f} → Predicted class: {pred_label}")

        except Exception as e:
            print(f"Error processing {f}: {e}")

    if results:
        # Ensure the CSV has the requested column order and columns present
        df = pd.DataFrame(results)
        cols = ['filename', 'predicted_label', 'predicted_confidence', 'actual_label']
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[cols]

        # Print header line as requested
        print('\n' + ','.join(cols))

        try:
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"\nPredictions saved to {OUTPUT_FILE}")
        except PermissionError:
            alt = f"result_{int(time.time())}.csv"
            try:
                df.to_csv(alt, index=False)
                print(f"\nPermission denied writing {OUTPUT_FILE}; saved to {alt} instead")
            except Exception as e:
                print(f"Failed to save predictions to fallback file: {e}")
        except Exception as e:
            print(f"Failed saving predictions: {e}")

    # print overall accuracy if we could infer true labels
    if total_count > 0:
        accuracy = correct_count / total_count
        print(f"\nAverage Accuracy: {accuracy*100:.2f}%")

    # If we collected actual labels, compute detailed sklearn metrics
    true_labels = [r['actual_label'] for r in results if r.get('actual_label') is not None]
    predicted_labels = [r['predicted_label'] for r in results if r.get('actual_label') is not None]

    if len(true_labels) > 0:
        y_true = np.array(true_labels)
        y_pred = np.array(predicted_labels)

        # Accuracy (sklearn)
        accuracy_skl = accuracy_score(y_true, y_pred)
        print(f"\nAverage Accuracy: {accuracy_skl * 100:.2f}%")

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        # Precision, Recall, F1-score (macro)
        report = classification_report(y_true, y_pred, digits=2)
        print("\nClassification Report:")
        print(report)


if __name__ == "__main__":
    main()
