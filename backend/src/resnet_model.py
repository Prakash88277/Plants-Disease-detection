"""
resnet_model.py
---------------
Builds, compiles, and trains the Stage-1 ResNet50 classifier.

Architecture
------------
  ResNet50 (ImageNet weights, base frozen)
  → GlobalAveragePooling2D
  → Dense(256, relu)
  → Dropout(0.3)
  → Dense(num_classes, softmax)

Training is done in two phases:
  Phase 1 – only the custom head is trained  (base frozen)
  Phase 2 – last 20 layers of ResNet50 are unfrozen and fine-tuned
             with a very small learning rate
"""

import os
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.applications import ResNet50
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
def build_resnet50(num_classes: int,
                   input_shape: tuple = (224, 224, 3),
                   freeze_base: bool  = True) -> Model:
    """
    Construct the ResNet50-based classifier.

    Parameters
    ----------
    num_classes  : int   – number of output classes (53 for plant disease)
    input_shape  : tuple – (H, W, C) expected by the model
    freeze_base  : bool  – if True, all ResNet50 base layers are frozen

    Returns
    -------
    keras.Model (compiled)
    """

    # Load pretrained ResNet50; exclude the top FC layers
    base_model = ResNet50(
        include_top  = False,
        weights      = "imagenet",
        input_shape  = input_shape,
    )
    base_model.trainable = not freeze_base   # freeze or unfreeze

    # ── Custom classification head ──
    inputs  = Input(shape=input_shape)
    x       = base_model(inputs, training=False)  # BN in inference mode when frozen
    x       = GlobalAveragePooling2D()(x)
    x       = Dense(256, activation="relu")(x)
    x       = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="ResNet50_PlantDisease")

    # Compile for Phase-1 training
    model.compile(
    optimizer = Adam(learning_rate=1e-3),  # HIGH LR
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"],
)

    print(f"[resnet_model] Built ResNet50 | classes={num_classes} | "
          f"base_frozen={freeze_base}")
    model.summary(line_length=80)

    return model


# ─────────────────────────────────────────────
# 2. Fine-tuning helper
# ─────────────────────────────────────────────
def unfreeze_last_n_layers(model: Model,
                            n_layers: int = 20,
                            new_lr: float = 1e-5) -> Model:
    """
    Unfreeze the last `n_layers` of the ResNet50 base for fine-tuning.

    Parameters
    ----------
    model    : compiled keras.Model returned by build_resnet50()
    n_layers : int   – how many layers to unfreeze from the end
    new_lr   : float – smaller learning rate for fine-tuning

    Returns
    -------
    The same model with some layers unfrozen and re-compiled.
    """
    # Locate the ResNet50 base (layer 0 is the Input, layer 1 is the base)
    base_model = model.layers[1]            # ResNet50 sub-model

    # Unfreeze selectively
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False             # keep earlier layers frozen

    # Re-compile with lower LR
    model.compile(
        optimizer = Adam(learning_rate=new_lr),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"[resnet_model] Fine-tuning: {trainable_count} / "
          f"{len(base_model.layers)} base layers trainable | lr={new_lr}")

    return model


# ─────────────────────────────────────────────
# 3. Standard callbacks
# ─────────────────────────────────────────────
def get_callbacks(checkpoint_path: str,
                  monitor: str   = "val_accuracy",
                  patience: int  = 7) -> list:
    """
    Return a list of training callbacks.

    Includes:
      - EarlyStopping      : stops when val_accuracy stops improving
      - ModelCheckpoint    : saves the best weights
      - ReduceLROnPlateau  : halves LR after 3 stagnant epochs

    Parameters
    ----------
    checkpoint_path : str – file path to save the best model  (.h5 or SavedModel)
    monitor         : str – metric to watch
    patience        : int – epochs to wait before early stopping

    Returns
    -------
    list of keras callbacks
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    early_stop = EarlyStopping(
        monitor              = monitor,
        patience             = patience,
        restore_best_weights = True,
        verbose              = 1,
    )

    checkpoint = ModelCheckpoint(
        filepath         = checkpoint_path,
        monitor           = "val_accuracy",
        save_best_only   = True,
        save_weights_only= False,
        mode              = "max",
        verbose          = 1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor   = "val_loss",
        factor    = 0.3,
        patience  = 2,
        min_lr    = 1e-7,
        verbose   = 1,
    )

    return [early_stop, checkpoint, reduce_lr]


# ─────────────────────────────────────────────
# 4. Full training function
# ─────────────────────────────────────────────
def train_resnet(train_gen,
                 val_gen,
                 num_classes: int,
                 save_path:   str   = "models/resnet_model.h5",
                 epochs_phase1: int = 20,
                 epochs_phase2: int = 10) -> tuple:
    """
    Two-phase training for ResNet50.

    Phase 1 – head only (base frozen)      : up to epochs_phase1 epochs
    Phase 2 – fine-tune last 20 base layers: up to epochs_phase2 epochs

    Parameters
    ----------
    train_gen      : Keras generator – training data
    val_gen        : Keras generator – validation data
    num_classes    : int             – number of plant-disease classes
    save_path      : str             – where to checkpoint the best model
    epochs_phase1  : int             – max epochs for phase 1
    epochs_phase2  : int             – max epochs for phase 2

    Returns
    -------
    (model, history_phase1, history_phase2)
    """

    # ── Phase 1 ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ResNet50  |  Phase 1 – training head only")
    print("="*60)

    model     = build_resnet50(num_classes=num_classes, freeze_base=True)
    callbacks = get_callbacks(save_path)

    history1  = model.fit(
        train_gen,
        validation_data  = val_gen,
        epochs           = epochs_phase1,
        callbacks        = callbacks,
        verbose          = 1,
    )

    # ── Phase 2 ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ResNet50  |  Phase 2 – fine-tuning last 20 layers")
    print("="*60)

    model     = unfreeze_last_n_layers(model, n_layers=20, new_lr=1e-5)
    callbacks = get_callbacks(save_path, patience=5)

    history2  = model.fit(
        train_gen,
        validation_data  = val_gen,
        epochs           = epochs_phase2,
        callbacks        = callbacks,
        verbose          = 1,
    )

    print(f"\n[resnet_model] Training complete. Best model saved → {save_path}")
    return model, history1, history2
