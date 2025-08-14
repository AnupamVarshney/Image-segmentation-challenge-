"""simple_unet.py
=================================
Simple UNet architecture for foreground/background segmentation from
sparse scribbles - basic version without attention/residual connections.

This is a simplified version of the enhanced UNet for comparison.
"""

import os
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from util import load_dataset, store_predictions, visualize
import time
import datetime

# Custom progress callback
class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.epoch_times = []
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"\n🚀 Simple UNet Training started at {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"📊 Total epochs: {EPOCHS}")
        print(f"🎯 Batch size: {BATCH}")
        print(f"⚡ Mixed precision: {MIXED_PRECISION}")
        print("=" * 60)
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
        print(f"\n📈 Epoch {epoch+1}/{EPOCHS} starting...")
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        avg_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = EPOCHS - (epoch + 1)
        eta = remaining_epochs * avg_time
        
        print(f"✅ Epoch {epoch+1}/{EPOCHS} completed in {epoch_time:.1f}s")
        print(f"   📊 Loss: {logs.get('loss', 0):.4f} | IoU: {logs.get('masked_iou', 0):.4f}")
        print(f"   🎯 Val Loss: {logs.get('val_loss', 0):.4f} | Val IoU: {logs.get('val_masked_iou', 0):.4f}")
        print(f"   ⏱️  Avg time/epoch: {avg_time:.1f}s | ETA: {eta/60:.1f}min")
        
        # Show progress bar
        progress = (epoch + 1) / EPOCHS * 100
        bar_length = 30
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"   📊 Progress: [{bar}] {progress:.1f}%")
        print("-" * 60)
        
    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print(f"\n🎉 Simple UNet Training completed in {total_time/60:.1f} minutes!")
        print(f"📈 Final metrics - Loss: {logs.get('loss', 0):.4f}, IoU: {logs.get('masked_iou', 0):.4f}")
        print(f"🎯 Best val IoU: {max(self.model.history.history.get('val_masked_iou', [0])):.4f}")

# ---------------------------------------------------------------------------
# 0.  Reproducibility --------------------------------------------------------
# ---------------------------------------------------------------------------
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# 1.  Configuration ----------------------------------------------------------
# ---------------------------------------------------------------------------
TARGET_SIZE = (375, 500)  # (h, w) final mask size for submission
PATCH       = 256         # training patch size
BATCH       = 16          # Batch size for A100 40GB
EPOCHS      = 12          # Reduced epochs for quick comparison
USE_PSEUDO  = False       # set True to train on Random‑Walker masks instead

# A100 Optimizations
MIXED_PRECISION = True    # Enable FP16 for ~2x speedup
NUM_PARALLEL_CALLS = 8    # Increase data loading parallelism

# A100 Mixed Precision Setup
if MIXED_PRECISION:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled for A100 optimization")

# ---------------------------------------------------------------------------
# 2.  Dataset loading --------------------------------------------------------
# ---------------------------------------------------------------------------
print("🔄 Loading dataset...")
images_train, scrib_train, gt_train, fnames_train, palette = load_dataset(
    "dataset/train", "images", "scribbles", "ground_truth")
images_test, _, fnames_test = load_dataset("dataset/test1", "images", "scribbles")

print(f"📊 Dataset loaded:")
print(f"   🖼️  Training images: {len(images_train)}")
print(f"   🖼️  Test images: {len(images_test)}")
print(f"   📏 Image shape: {images_train[0].shape}")
print(f"   🎨 Palette: {palette}")

# Normalise images to [0,1]
images_train = images_train.astype(np.float32) / 255.0
images_test  = images_test.astype(np.float32)  / 255.0

# ---------------------------------------------------------------------------
# 3.  Optional Random‑Walker pseudo‑labels -----------------------------------
# ---------------------------------------------------------------------------
if USE_PSEUDO:
    from skimage.segmentation import random_walker

    def make_rw_mask(img, scrib):
        if scrib.ndim == 3:
            scrib = scrib[..., 0]
        gray = (img[..., 0]*0.299 + img[..., 1]*0.587 + img[..., 2]*0.114)
        seeds = np.full_like(scrib, -1, dtype=np.int32)
        seeds[scrib == 0] = 0          # background
        seeds[scrib == 1] = 1          # foreground
        mask = random_walker(gray, seeds, beta=80, mode='bf')
        return (mask == 1).astype(np.uint8)

    print("Generating Random‑Walker masks …")
    rw_masks = np.stack([make_rw_mask(img, scr)
                         for img, scr in zip(images_train, scrib_train)], axis=0)
    mask_train = rw_masks[..., np.newaxis].astype(np.float32)
    print("Done.")
else:
    # use scribbles directly (0/1 labelled, 255 unknown)
    mask_train = scrib_train[..., np.newaxis].astype(np.float32)

# ---------------------------------------------------------------------------
# 4.  Utility – pad / unpad to multiple of 8 --------------------------------
# ---------------------------------------------------------------------------

def pad_to_multiple(arr, m=8, value=0):
    h, w = arr.shape[:2]
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph == 0 and pw == 0:
        return arr
    pad_spec = ((0, ph), (0, pw)) + (() if arr.ndim == 2 else ((0, 0),))
    return np.pad(arr, pad_spec, mode='constant', constant_values=value)

# Pad all training images & masks so they share identical shapes
images_train = np.array([pad_to_multiple(im) for im in images_train])
mask_train   = np.array([pad_to_multiple(msk) for msk in mask_train])

# ---------------------------------------------------------------------------
# 5.  Patch extraction -------------------------------------------------------
# ---------------------------------------------------------------------------

def np_random_patch(img, msk):
    """Return a PATCH×PATCH crop that contains ≥ some labelled pixels."""
    h, w = img.shape[:2]
    for _ in range(10):
        top  = np.random.randint(0, h-PATCH+1)
        left = np.random.randint(0, w-PATCH+1)
        patch_img = img[top:top+PATCH, left:left+PATCH]
        patch_msk = msk[top:top+PATCH, left:left+PATCH]
        # ensure at least ~50 labelled pixels
        lab = (patch_msk != 255) if not USE_PSEUDO else (patch_msk > 0)
        if lab.sum() > 50:
            return patch_img, patch_msk
    # fallback – return last crop even if mostly unlabeled
    return patch_img, patch_msk

def tf_random_patch(img, msk):
    img, msk = tf.numpy_function(np_random_patch, [img, msk], [tf.float32, tf.float32])
    img.set_shape([PATCH, PATCH, 3])
    msk.set_shape([PATCH, PATCH, 1])
    return img, msk

# ---------------------------------------------------------------------------
# 6.  Augmentations ----------------------------------------------------------
# ---------------------------------------------------------------------------

def augment(img, msk):
    if tf.random.uniform(()) > 0.5:
        img  = tf.image.flip_left_right(img)
        msk  = tf.image.flip_left_right(msk)
    if tf.random.uniform(()) > 0.5:
        img  = tf.image.flip_up_down(img)
        msk  = tf.image.flip_up_down(msk)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.random_brightness(img, 0.12)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.random_contrast(img, 0.9, 1.1)
    return img, msk

# Build tf.data pipeline
N = len(images_train)
val_split = int(0.15 * N)

full_ds = tf.data.Dataset.from_tensor_slices((images_train, mask_train))
train_ds = (full_ds.skip(val_split)
            .map(tf_random_patch, num_parallel_calls=NUM_PARALLEL_CALLS)
            .map(augment,          num_parallel_calls=NUM_PARALLEL_CALLS)
            .batch(BATCH)
            .prefetch(tf.data.AUTOTUNE))
val_ds   = (full_ds.take(val_split)
            .map(tf_random_patch, num_parallel_calls=NUM_PARALLEL_CALLS)
            .batch(BATCH)
            .prefetch(tf.data.AUTOTUNE))

# ---------------------------------------------------------------------------
# 7.  Simple UNet definition -------------------------------------------------
# ---------------------------------------------------------------------------

def simple_unet(input_shape=(None, None, 3)):
    """Simple U-Net without attention or residual connections"""
    inputs = keras.Input(shape=input_shape)
    
    # Encoder
    # Level 1
    c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    
    # Level 2
    c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(p1)
    c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    
    # Level 3
    c3 = layers.Conv2D(256, 3, padding='same', activation='relu')(p2)
    c3 = layers.Conv2D(256, 3, padding='same', activation='relu')(c3)
    p3 = layers.MaxPooling2D(2)(c3)
    
    # Level 4
    c4 = layers.Conv2D(512, 3, padding='same', activation='relu')(p3)
    c4 = layers.Conv2D(512, 3, padding='same', activation='relu')(c4)
    p4 = layers.MaxPooling2D(2)(c4)
    
    # Bottleneck
    bottleneck = layers.Conv2D(1024, 3, padding='same', activation='relu')(p4)
    bottleneck = layers.Conv2D(1024, 3, padding='same', activation='relu')(bottleneck)
    
    # Decoder with proper size handling
    # Level 4 decode
    u4 = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(bottleneck)
    # Ensure same size for concatenation
    u4_size = tf.shape(u4)[1:3]
    c4_resized = tf.image.resize(c4, u4_size, method='nearest')
    u4 = layers.Concatenate()([u4, c4_resized])
    c5 = layers.Conv2D(512, 3, padding='same', activation='relu')(u4)
    c5 = layers.Conv2D(512, 3, padding='same', activation='relu')(c5)
    
    # Level 3 decode
    u3 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c5)
    u3_size = tf.shape(u3)[1:3]
    c3_resized = tf.image.resize(c3, u3_size, method='nearest')
    u3 = layers.Concatenate()([u3, c3_resized])
    c6 = layers.Conv2D(256, 3, padding='same', activation='relu')(u3)
    c6 = layers.Conv2D(256, 3, padding='same', activation='relu')(c6)
    
    # Level 2 decode
    u2 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c6)
    u2_size = tf.shape(u2)[1:3]
    c2_resized = tf.image.resize(c2, u2_size, method='nearest')
    u2 = layers.Concatenate()([u2, c2_resized])
    c7 = layers.Conv2D(128, 3, padding='same', activation='relu')(u2)
    c7 = layers.Conv2D(128, 3, padding='same', activation='relu')(c7)
    
    # Level 1 decode
    u1 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c7)
    u1_size = tf.shape(u1)[1:3]
    c1_resized = tf.image.resize(c1, u1_size, method='nearest')
    u1 = layers.Concatenate()([u1, c1_resized])
    c8 = layers.Conv2D(64, 3, padding='same', activation='relu')(u1)
    c8 = layers.Conv2D(64, 3, padding='same', activation='relu')(c8)
    
    # Final output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c8)
    
    model = keras.Model(inputs, outputs)
    return model

# Create the simple model
model = simple_unet()
print("Simple U-Net Architecture:")
model.summary(line_length=120)

# ---------------------------------------------------------------------------
# 8.  Loss & metrics ---------------------------------------------------------
# ---------------------------------------------------------------------------

def masked_bce(y_t, y_p):
    mask = tf.not_equal(y_t, 255.0)
    y_t  = tf.boolean_mask(y_t, mask)
    y_p  = tf.boolean_mask(y_p, mask)
    # class weighting
    pos = tf.reduce_sum(y_t)
    neg = tf.reduce_sum(1.0 - y_t)
    alpha = neg / (pos + neg + 1e-6)
    beta  = pos / (pos + neg + 1e-6)
    w = y_t * alpha + (1 - y_t) * beta
    bce = tf.keras.backend.binary_crossentropy(y_t, y_p)
    return tf.reduce_mean(w * bce)

def masked_dice(y_t, y_p, smooth=1e-6):
    mask = tf.not_equal(y_t, 255.0)
    y_t  = tf.boolean_mask(y_t, mask)
    y_p  = tf.boolean_mask(y_p, mask)
    inter = tf.reduce_sum(y_t * y_p)
    return 1 - (2*inter + smooth) / (tf.reduce_sum(y_t) + tf.reduce_sum(y_p) + smooth)

def total_loss(y_t, y_p):
    return masked_bce(y_t, y_p) + masked_dice(y_t, y_p)

def masked_iou(y_t, y_p):
    mask = tf.not_equal(y_t, 255.0)
    y_t  = tf.boolean_mask(y_t, mask)
    y_p  = tf.cast(tf.boolean_mask(y_p, mask) > 0.5, tf.float32)
    inter = tf.reduce_sum(y_t * y_p)
    union = tf.reduce_sum(y_t) + tf.reduce_sum(y_p) - inter
    return inter / (union + 1e-6)

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    ),
    loss=total_loss,
    metrics=[masked_iou]
)

# Callbacks
callbacks = [
    ReduceLROnPlateau(
        monitor='val_masked_iou', 
        mode='max', 
        factor=0.5,
        patience=4,
        verbose=1, 
        min_lr=1e-7
    ),
    EarlyStopping(
        monitor='val_masked_iou', 
        mode='max', 
        patience=10,
        restore_best_weights=True, 
        verbose=1
    ),
    ProgressCallback()
]

# Train the simple model
print(f"Training Simple UNet on A100 with batch size {BATCH}, mixed precision: {MIXED_PRECISION}")

# GPU memory optimization
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s), memory growth enabled")
    except RuntimeError as e:
        print(e)

model.fit(
    train_ds, 
    epochs=EPOCHS, 
    validation_data=val_ds,
    callbacks=callbacks, 
    verbose=2
)

# ---------------------------------------------------------------------------
# 9.  Prediction & saving ----------------------------------------------------
# ---------------------------------------------------------------------------
THRESH = 0.4

def predict_mask(img):
    """Predict binary mask for a single RGB uint8 image using simple U-Net."""
    h0, w0, _ = img.shape
    img_f = img.astype(np.float32) / 255.0
    pad_img = pad_to_multiple(img_f)
    
    # Model prediction
    pred = model.predict(pad_img[np.newaxis])[0, ..., 0]
    
    # Crop back to original size
    pr = pred[:h0, :w0]
    pr_bin = (pr >= THRESH).astype(np.uint8)
    
    # Morphological cleaning
    kernel = np.ones((3,3), np.uint8)
    pr_clean = cv2.morphologyEx(pr_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    pr_clean = cv2.morphologyEx(pr_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # resize/crop to target 375×500
    pr_resized = cv2.resize(pr_clean, (TARGET_SIZE[1], TARGET_SIZE[0]),
                            interpolation=cv2.INTER_NEAREST)
    return pr_resized

print("Saving Simple UNet predictions …")
train_preds = np.stack([predict_mask((im*255).astype(np.uint8))
                        for im in images_train], axis=0)
store_predictions(train_preds, "dataset/train", "simple_unet", fnames_train, palette)

test_preds = np.stack([predict_mask((im*255).astype(np.uint8))
                       for im in images_test], axis=0)
store_predictions(test_preds, "dataset/test1", "simple_unet", fnames_test, palette)
print("Done.")

# ---------------------------------------------------------------------------
# 10. Quick qualitative check ----------------------------------------------
# ---------------------------------------------------------------------------
idx = np.random.randint(len(images_train))
padded_scrib = pad_to_multiple(scrib_train[idx])
visualize((images_train[idx]*255).astype(np.uint8),
          padded_scrib,
          pad_to_multiple(gt_train[idx]),
          train_preds[idx]) 