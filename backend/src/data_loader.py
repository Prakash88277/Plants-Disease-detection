"""
data_loader.py
--------------
Handles all data loading, augmentation, and preprocessing for
the plant disease detection pipeline.

Dataset structure expected:
    data/
        Apple___Black_rot/
        Apple___healthy/
        Tomato___Early_blight/
        ... (53 total classes)
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 32
VALID_SPLIT  = 0.2          # 80/20 train/val split
RANDOM_SEED  = 42


# ─────────────────────────────────────────────
# 1. Data generators (train + validation)
# ─────────────────────────────────────────────
def get_data_generators(data_dir: str,
                         image_size: tuple = IMAGE_SIZE,
                         batch_size: int   = BATCH_SIZE,
                         valid_split: float = VALID_SPLIT):
    """
    Build ImageDataGenerator pipelines for training and validation.

    Augmentation applied only to training set:
      - Random rotation ±20°
      - Width / height shift ±10 %
      - Zoom ±15 %
      - Horizontal flip
      - Brightness jitter 80–120 %
      - Nearest-pixel fill for empty corners

    Parameters
    ----------
    data_dir   : str   – root folder that contains the 53 class sub-folders
    image_size : tuple – (height, width) fed to the networks  (default 224×224)
    batch_size : int   – mini-batch size                       (default 32)
    valid_split: float – fraction held out for validation      (default 0.20)

    Returns
    -------
    train_generator, val_generator, class_names
    """

    # -- Training generator (with augmentation, NO rescaling for EfficientNet) --
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,   # ✅ FIX
        rotation_range     = 30,
        width_shift_range  = 0.15,
        height_shift_range = 0.15,
        shear_range        = 0.15,
        zoom_range         = 0.2,
        vertical_flip      = True,
        horizontal_flip    = True,
        brightness_range=[0.9, 1.1],   # ✅ reduce intensity
        # brightness_range   = [0.8, 1.2],
        fill_mode          = "nearest",
        validation_split   = valid_split,
    )

    # # -- Validation generator (NO rescaling) --
    # val_datagen = ImageDataGenerator(
    #     validation_split = valid_split,
    # )


    # -- Training generator --
    # train_datagen = ImageDataGenerator(
    #     rotation_range=20,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     zoom_range=0.15,
    #     horizontal_flip=True,
    #     fill_mode="nearest",
    #     validation_split=valid_split,
    # )

    # -- Validation generator --
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,   # ✅ FIX
        validation_split=valid_split,
    )

    # -- Flow from directory --
    train_generator = train_datagen.flow_from_directory(
        directory    = data_dir,
        target_size  = image_size,
        batch_size   = batch_size,
        class_mode   = "categorical",
        subset       = "training",
        shuffle      = True,
        seed         = RANDOM_SEED,
    )

    val_generator = val_datagen.flow_from_directory(
        directory    = data_dir,
        target_size  = image_size,
        batch_size   = batch_size,
        class_mode   = "categorical",
        subset       = "validation",
        shuffle      = False,
        seed         = RANDOM_SEED,
    )

    class_names = list(train_generator.class_indices.keys())
    print(f"[data_loader] Found {len(class_names)} classes.")
    print(f"[data_loader] Training   samples : {train_generator.samples}")
    print(f"[data_loader] Validation samples : {val_generator.samples}")

    return train_generator, val_generator, class_names


# ─────────────────────────────────────────────
# 2. Single-image preprocessing (for inference)
# ─────────────────────────────────────────────
def preprocess_image(image_path: str,
                     image_size: tuple = IMAGE_SIZE) -> np.ndarray:
    """
    Load one image from disk and prepare it for model inference.

    Steps:
      1. Load as RGB PIL image, resize to image_size
      2. Convert to float32 numpy array  (H, W, 3)
      3. Do NOT rescale (EfficientNet handles [0-255] internally)
      4. Add batch dimension              (1, H, W, 3)

    Parameters
    ----------
    image_path : str   – path to the image file
    image_size : tuple – target (height, width)

    Returns
    -------
    np.ndarray of shape (1, H, W, 3)
    """
    img   = load_img(image_path, target_size=image_size)
    arr   = img_to_array(img).astype("float32") # Removed / 255.0
    batch = np.expand_dims(arr, axis=0)          # add batch dim
    return batch


# ─────────────────────────────────────────────
# 3. Dataset statistics helper
# ─────────────────────────────────────────────
def dataset_summary(data_dir: str) -> dict:
    """
    Print and return basic statistics about the dataset folder.

    Parameters
    ----------
    data_dir : str – root folder with class sub-folders

    Returns
    -------
    dict with keys: num_classes, class_names, class_counts, total_images
    """
    class_names  = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    class_counts = {}
    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        files   = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        class_counts[cls] = len(files)

    total = sum(class_counts.values())

    print(f"\n{'='*50}")
    print(f"  Dataset Summary")
    print(f"{'='*50}")
    print(f"  Root         : {data_dir}")
    print(f"  Classes      : {len(class_names)}")
    print(f"  Total images : {total}")
    print(f"{'='*50}\n")

    return {
        "num_classes"  : len(class_names),
        "class_names"  : class_names,
        "class_counts" : class_counts,
        "total_images" : total,
    }
