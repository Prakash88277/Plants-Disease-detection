"""
utils.py
--------
Shared utilities for the plant disease detection project:

  • parse_class_name()      – split "Tomato___Early_blight" into plant + disease
  • predict()               – full cascaded inference pipeline
  • plot_training_history() – accuracy & loss curves
  • plot_confusion_matrix() – pretty confusion matrix heatmap
  • compute_gradcam()       – Grad-CAM saliency map
  • overlay_gradcam()       – blend Grad-CAM heatmap onto original image
  • show_gradcam()          – end-to-end Grad-CAM visualisation helper
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report


# ─────────────────────────────────────────────
# 1. Class-name parser
# ─────────────────────────────────────────────
def parse_class_name(class_name: str) -> tuple[str, str]:
    """
    Split a combined class label into plant and disease components.

    Examples
    --------
    >>> parse_class_name("Tomato___Early_blight")
    ('Tomato', 'Early blight')
    >>> parse_class_name("Apple___healthy")
    ('Apple', 'Healthy')
    >>> parse_class_name("Grape___Black_rot")
    ('Grape', 'Black rot')

    Parameters
    ----------
    class_name : str – folder/class name in the dataset

    Returns
    -------
    (plant, disease) both as human-readable strings
    """
    parts   = class_name.split("___", maxsplit=1)
    plant   = parts[0].strip()
    disease = parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown"
    return plant, disease


# ─────────────────────────────────────────────
# 2. Cascaded inference pipeline
# ─────────────────────────────────────────────
def predict(image_path:      str,
            resnet_model,
            efficientnet_model,
            class_names:     list,
            threshold:       float = 0.30,
            image_size:      tuple = (224, 224)) -> dict:
    """
    Full cascaded inference: ResNet50 → EfficientNetB0.

    Logic
    -----
    1. Preprocess image (resize, normalise, add batch dim)
    2. Run ResNet50 → extract top-1 confidence
    3. If confidence >= threshold  → pass to EfficientNetB0
       Else                        → still pass, but flag as low-confidence
    4. Final prediction is always the EfficientNetB0 output
    5. Parse the winning class into (plant, disease)
    6. Return structured result dict

    Parameters
    ----------
    image_path          : str   – path to the leaf image
    resnet_model        : keras.Model – loaded ResNet50 model
    efficientnet_model  : keras.Model – loaded EfficientNetB0 model
    class_names         : list  – ordered list of 53 class labels
    threshold           : float – minimum ResNet confidence to accept (default 0.30)
    image_size          : tuple – (H, W) fed to both models

    Returns
    -------
    dict with keys:
        plant, disease, confidence (0–1),
        predicted_class, stage1_confidence,
        high_confidence (bool), all_probs
    """

    # ── 1. Preprocess ───────────────────────────────────────────────────
    img  = load_img(image_path, target_size=image_size)
    arr_raw = img_to_array(img).astype("float32")
    
    # ResNet50 was trained WITH 1.0/255.0 scaling
    inp_resnet = np.expand_dims(arr_raw / 255.0, axis=0)
    
    # EfficientNetB0 was trained WITHOUT manual scaling (expects [0, 255])
    inp_eff = np.expand_dims(arr_raw, axis=0)

    # ── 2. Stage 1: ResNet50 ────────────────────────────────────────────
    stage1_probs     = resnet_model.predict(inp_resnet, verbose=0)[0]
    stage1_top_conf  = float(np.max(stage1_probs))
    stage1_top_cls   = class_names[int(np.argmax(stage1_probs))]

    # ── 3. Filtering logic ──────────────────────────────────────────────
    high_confidence = stage1_top_conf >= threshold
    status_msg = "✓ High confidence" if high_confidence else "⚠ Low confidence – still forwarding"
    print(f"[predict] Stage 1 ({stage1_top_cls}) conf={stage1_top_conf:.3f}  {status_msg}")

    # ── 4. Stage 2: EfficientNetB0 ──────────────────────────────────────
    stage2_probs   = efficientnet_model.predict(inp_eff, verbose=0)[0]
    
    # Debug raw predictions
    print("Raw predictions max prob:", np.max(stage2_probs))
    print("Predicted class index:", np.argmax(stage2_probs))
    
    top_idx        = int(np.argmax(stage2_probs))
    predicted_cls  = class_names[top_idx]
    confidence     = float(stage2_probs[top_idx])

    # ── 5. Parse class label ────────────────────────────────────────────
    plant, disease = parse_class_name(predicted_cls)

    if confidence < 0.3:
        disease = "Low confidence"
        print(f"[predict] Stage 2 → Plant: {plant} | Disease: {disease} | Confidence: {confidence:.3f}")
        return {
            "plant": plant,
            "disease": disease,
            "confidence": round(confidence, 4),
            "predicted_class": predicted_cls,
            "stage1_confidence": round(stage1_top_conf, 4),
            "high_confidence": high_confidence,
            "all_probs": stage2_probs,
        }

    # ── 6. Build result ─────────────────────────────────────────────────
    result = {
        "plant"            : plant,
        "disease"          : disease,
        "confidence"       : round(confidence, 4),
        "predicted_class"  : predicted_cls,
        "stage1_confidence": round(stage1_top_conf, 4),
        "high_confidence"  : high_confidence,
        "all_probs"        : stage2_probs,      # numpy array
    }

    print(f"[predict] Stage 2 → Plant: {plant} | Disease: {disease} | "
          f"Confidence: {confidence:.3f}")

    return result


# ─────────────────────────────────────────────
# 3. Training history visualisation
# ─────────────────────────────────────────────
def plot_training_history(history1,
                          history2=None,
                          model_name: str = "Model",
                          save_path: str  = None):
    """
    Plot accuracy and loss curves for one or two training phases.

    Parameters
    ----------
    history1   : keras History object – Phase 1 results
    history2   : keras History object – Phase 2 results (optional)
    model_name : str – used in the plot title
    save_path  : str – if given, figure is saved to this path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} – Training History", fontsize=14, fontweight="bold")

    # Merge histories if two phases exist
    def _merge(h1, h2, key):
        vals = h1.history.get(key, [])
        if h2 is not None:
            vals = vals + h2.history.get(key, [])
        return vals

    epochs     = range(1, len(_merge(history1, history2, "accuracy")) + 1)
    p1_end     = len(history1.history.get("accuracy", []))

    # ── Accuracy ──
    ax = axes[0]
    ax.plot(epochs, _merge(history1, history2, "accuracy"),     label="Train Accuracy", color="#2196F3")
    ax.plot(epochs, _merge(history1, history2, "val_accuracy"), label="Val Accuracy",   color="#FF5722")
    if history2:
        ax.axvline(p1_end, linestyle="--", color="gray", alpha=0.6, label="Phase 1 → 2")
    ax.set_title("Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(alpha=0.3)

    # ── Loss ──
    ax = axes[1]
    ax.plot(epochs, _merge(history1, history2, "loss"),         label="Train Loss",  color="#4CAF50")
    ax.plot(epochs, _merge(history1, history2, "val_loss"),     label="Val Loss",    color="#9C27B0")
    if history2:
        ax.axvline(p1_end, linestyle="--", color="gray", alpha=0.6, label="Phase 1 → 2")
    ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[utils] History plot saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# 4. Confusion matrix
# ─────────────────────────────────────────────
def plot_confusion_matrix(model,
                          val_generator,
                          class_names: list,
                          save_path:   str = None,
                          figsize:     tuple = (20, 18)):
    """
    Generate predictions on the validation set and plot a confusion matrix.

    Parameters
    ----------
    model         : keras.Model – trained classifier
    val_generator : Keras generator – validation data (shuffle=False)
    class_names   : list of str
    save_path     : str – optional path to save the figure
    figsize       : tuple – matplotlib figure size
    """
    # Collect ground-truth labels and predictions
    val_generator.reset()
    y_true, y_pred = [], []

    print("[utils] Running predictions for confusion matrix …")
    for i, (imgs, labels) in enumerate(val_generator):
        preds  = model.predict(imgs, verbose=0)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(preds,  axis=1))
        if (i + 1) * val_generator.batch_size >= val_generator.samples:
            break

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot    = True,
        fmt      = "d",
        cmap     = "Blues",
        xticklabels = class_names,
        yticklabels = class_names,
        ax       = ax,
        linewidths  = 0.3,
    )
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[utils] Confusion matrix saved → {save_path}")
    plt.show()

    # Also print text report
    print("\n" + classification_report(
        y_true, y_pred,
        target_names = class_names,
        digits       = 3,
    ))
