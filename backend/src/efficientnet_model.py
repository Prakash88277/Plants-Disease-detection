"""
efficientnet_model.py
---------------------
Builds, compiles, and trains the Stage-2 EfficientNetB0 classifier.

Architecture
------------
  EfficientNetB0 (ImageNet weights, base frozen)
  → GlobalAveragePooling2D
  → Dense(256, relu)
  → Dropout(0.3)
  → Dense(num_classes, softmax)

Mirrors the ResNet50 training recipe:
  Phase 1 – custom head only (base frozen)
  Phase 2 – last 20 layers unfrozen and fine-tuned at a low LR
"""

import os
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)


# ─────────────────────────────────────────────
# 1. Model builder
# ─────────────────────────────────────────────
def build_efficientnet(num_classes: int,
                       input_shape: tuple = (224, 224, 3),
                       freeze_base: bool  = True) -> Model:
    """
    Construct the EfficientNetB0-based classifier.

    Note: EfficientNet includes its own internal rescaling layer, so
    pixel values should still be in [0, 1] (as produced by the
    ImageDataGenerator rescale=1/255 setting).

    Parameters
    ----------
    num_classes  : int   – number of output classes (53 for plant disease)
    input_shape  : tuple – (H, W, C)
    freeze_base  : bool  – freeze base layers when True

    Returns
    -------
    keras.Model (compiled)
    """

    base_model = EfficientNetB0(
        include_top  = False,
        weights      = "imagenet",
        input_shape  = input_shape,
    )
    base_model.trainable = not freeze_base

    # ── Custom head ──
    inputs  = Input(shape=input_shape)
    x       = base_model(inputs, training=False)
    x       = GlobalAveragePooling2D()(x)
    x       = Dense(256, activation="relu")(x)
    x       = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="EfficientNetB0_PlantDisease")

    model.compile(
        optimizer = Adam(learning_rate=1e-3),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    print(f"[efficientnet_model] Built EfficientNetB0 | classes={num_classes} | "
          f"base_frozen={freeze_base}")
    model.summary(line_length=80)

    return model


# ─────────────────────────────────────────────
# 2. Fine-tuning helper
# ─────────────────────────────────────────────
def unfreeze_last_n_layers(model: Model,
                            n_layers: int = 60,
                            new_lr: float = 1e-4) -> Model:
    """
    Unfreeze the last `n_layers` of EfficientNetB0 for fine-tuning.

    Parameters
    ----------
    model    : compiled keras.Model returned by build_efficientnet()
    n_layers : int   – layers to unfreeze from the end of the base
    new_lr   : float – fine-tuning learning rate

    Returns
    -------
    Re-compiled model
    """
    base_model = model.layers[1]           # EfficientNetB0 sub-model

    base_model.trainable = True
    
    # Freeze all layers EXCEPT the last n_layers
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False
        
    # Also explicitly freeze all BatchNormalization layers in the unfrozen part
    for layer in base_model.layers[-n_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer = Adam(learning_rate=new_lr),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"[efficientnet_model] Fine-tuning: {trainable_count} / "
          f"{len(base_model.layers)} base layers trainable | lr={new_lr}")

    return model


# ─────────────────────────────────────────────
# 3. Callbacks (identical to resnet_model.py)
# ─────────────────────────────────────────────
def get_callbacks(checkpoint_path: str,
                  monitor: str  = "val_accuracy",
                  patience: int = 7) -> list:
    """
    Standard callback stack: EarlyStopping + ModelCheckpoint +
    ReduceLROnPlateau.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    early_stop = EarlyStopping(
        monitor              = monitor,
        patience             = patience,
        restore_best_weights = True,
        verbose              = 1,
    )

    checkpoint = ModelCheckpoint(
        filepath          = checkpoint_path,
        monitor           = monitor,
        save_best_only    = True,
        save_weights_only = False,
        verbose           = 1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor  = "val_loss",
        factor   = 0.5,
        patience = 3,
        min_lr   = 1e-7,
        verbose  = 1,
    )

    return [early_stop, checkpoint, reduce_lr]


# ─────────────────────────────────────────────
# 4. Full two-phase training function
# ─────────────────────────────────────────────
def train_efficientnet(train_gen,
                       val_gen,
                       num_classes:    int = 53,
                       save_path:      str = "models/efficientnet_model.h5",
                       epochs_phase1:  int = 20,
                       epochs_phase2:  int = 10) -> tuple:
    """
    Two-phase training for EfficientNetB0.

    Parameters
    ----------
    train_gen      : Keras generator
    val_gen        : Keras generator
    num_classes    : int – plant-disease class count
    save_path      : str – checkpoint destination
    epochs_phase1  : int – max epochs for Phase 1
    epochs_phase2  : int – max epochs for Phase 2

    Returns
    -------
    (model, history_phase1, history_phase2)
    """

    # ── Phase 1 ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  EfficientNetB0  |  Phase 1 – training head only")
    print("="*60)

    path_dir = os.path.dirname(save_path)
    base_name = os.path.basename(save_path)
    name, ext = os.path.splitext(base_name)
    phase1_path = os.path.join(path_dir, f"{name}_phase1{ext}")
    phase2_path = os.path.join(path_dir, f"{name}_phase2{ext}")

    model     = build_efficientnet(num_classes=num_classes, freeze_base=True)
    callbacks = get_callbacks(phase1_path)

    history1  = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = epochs_phase1,
        callbacks       = callbacks,
        verbose         = 1,
    )

    # Load best weights from phase 1 before unfreezing
    model.load_weights(phase1_path)

    # ── Phase 2 ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  EfficientNetB0  |  Phase 2 – fine-tuning top layers")
    print("="*60)

    model     = unfreeze_last_n_layers(model, n_layers=60, new_lr=1e-4)
    callbacks = get_callbacks(phase2_path, patience=5)

    history2  = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = epochs_phase2,
        callbacks       = callbacks,
        verbose         = 1,
    )

    print(f"\n[efficientnet_model] Training complete. Best model → {save_path}")
    return model, history1, history2
