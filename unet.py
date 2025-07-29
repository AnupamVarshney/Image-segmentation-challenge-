"""unet_segmentation_v5.py
=================================
End‑to‑end training script for foreground/background segmentation from
sparse scribbles **without** any external pre‑training.

Key changes vs. previous versions
---------------------------------
1. **Patch‑based training** – each batch now contains 256×256 crops that are
   guaranteed to include labelled pixels ⇒ far richer supervision.
2. **Masked, class‑balanced BCE + Dice loss** – computed only on pixels that
   are not 255.
3. **Masked IoU metric** – drives LR schedule / early stopping without being
   polluted by unlabeled pixels.
4. **Clean UNet** – BatchNorm + Dropout + Conv2DTranspose upsampling.
5. **Prediction pipeline** – resizes / pads back to the original size, then
   optional morphology to remove salt‑and‑pepper noise and finally saves a
   500 × 375 mask, as required by the challenge.

If you still want to experiment with Random‑Walker pseudo‑labels, set
`USE_PSEUDO = True` and the code will switch automatically.
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
BATCH       = 4
EPOCHS      = 120
USE_PSEUDO  = False       # set True to train on Random‑Walker masks instead

# ---------------------------------------------------------------------------
# 2.  Dataset loading --------------------------------------------------------
# ---------------------------------------------------------------------------
images_train, scrib_train, gt_train, fnames_train, palette = load_dataset(
    "dataset/train", "images", "scribbles", "ground_truth")
images_test, _, fnames_test = load_dataset("dataset/test1", "images", "scribbles")

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
            .map(tf_random_patch, num_parallel_calls=tf.data.AUTOTUNE)
            .map(augment,          num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH)
            .prefetch(tf.data.AUTOTUNE))
val_ds   = (full_ds.take(val_split)
            .map(tf_random_patch, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH)
            .prefetch(tf.data.AUTOTUNE))

# ---------------------------------------------------------------------------
# 7.  Enhanced UNet definition with Attention and Residual Connections -----
# ---------------------------------------------------------------------------

def squeeze_excite_block(x, ratio=16):
    """Squeeze-and-Excitation block for channel attention"""
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])

def attention_gate(g, x):
    """Attention gate for skip connections"""
    g_filters = g.shape[-1]
    x_filters = x.shape[-1]
    
    g_conv = layers.Conv2D(g_filters, 1, padding='same')(g)
    x_conv = layers.Conv2D(g_filters, 1, padding='same')(x)
    
    add = layers.Add()([g_conv, x_conv])
    relu = layers.ReLU()(add)
    
    psi = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(relu)
    
    return layers.Multiply()([x, psi])

def residual_conv_block(x, filters, dropout_rate=0.1):
    """Residual convolution block with squeeze-excitation"""
    shortcut = x
    
    # First conv block
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Second conv block
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    # Squeeze-and-Excitation
    x = squeeze_excite_block(x)
    
    # Residual connection
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    return x

def enhanced_unet(input_shape=(None, None, 3)):
    """Enhanced U-Net with attention, residual connections, and deeper architecture"""
    inputs = keras.Input(shape=input_shape)
    
    # Encoder with increasing filters and residual blocks
    # Level 1
    c1 = residual_conv_block(inputs, 64, 0.05)
    c1 = residual_conv_block(c1, 64, 0.05)
    p1 = layers.MaxPooling2D(2)(c1)
    
    # Level 2
    c2 = residual_conv_block(p1, 128, 0.1)
    c2 = residual_conv_block(c2, 128, 0.1)
    p2 = layers.MaxPooling2D(2)(c2)
    
    # Level 3
    c3 = residual_conv_block(p2, 256, 0.15)
    c3 = residual_conv_block(c3, 256, 0.15)
    p3 = layers.MaxPooling2D(2)(c3)
    
    # Level 4
    c4 = residual_conv_block(p3, 512, 0.2)
    c4 = residual_conv_block(c4, 512, 0.2)
    p4 = layers.MaxPooling2D(2)(c4)
    
    # Bottleneck with ASPP (Atrous Spatial Pyramid Pooling)
    # Multiple dilated convolutions to capture multi-scale context
    aspp1 = layers.Conv2D(1024, 1, padding='same', activation='relu')(p4)
    aspp6 = layers.Conv2D(1024, 3, padding='same', dilation_rate=6, activation='relu')(p4)
    aspp12 = layers.Conv2D(1024, 3, padding='same', dilation_rate=12, activation='relu')(p4)
    aspp18 = layers.Conv2D(1024, 3, padding='same', dilation_rate=18, activation='relu')(p4)
    
    # Simplified global context - just use the ASPP branches without global pooling
    bottleneck = layers.Concatenate()([aspp1, aspp6, aspp12, aspp18])
    bottleneck = layers.Conv2D(1024, 1, padding='same', activation='relu')(bottleneck)
    bottleneck = layers.BatchNormalization()(bottleneck)
    bottleneck = layers.Dropout(0.3)(bottleneck)
    
    # Decoder with attention gates and skip connections
    # Level 4 decode
    u4 = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(bottleneck)
    att4 = attention_gate(u4, c4)
    u4 = layers.Concatenate()([u4, att4])
    c5 = residual_conv_block(u4, 512, 0.2)
    c5 = residual_conv_block(c5, 512, 0.2)
    
    # Level 3 decode
    u3 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c5)
    att3 = attention_gate(u3, c3)
    u3 = layers.Concatenate()([u3, att3])
    c6 = residual_conv_block(u3, 256, 0.15)
    c6 = residual_conv_block(c6, 256, 0.15)
    
    # Level 2 decode
    u2 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c6)
    att2 = attention_gate(u2, c2)
    u2 = layers.Concatenate()([u2, att2])
    c7 = residual_conv_block(u2, 128, 0.1)
    c7 = residual_conv_block(c7, 128, 0.1)
    
    # Level 1 decode
    u1 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c7)
    att1 = attention_gate(u1, c1)
    u1 = layers.Concatenate()([u1, att1])
    c8 = residual_conv_block(u1, 64, 0.05)
    c8 = residual_conv_block(c8, 64, 0.05)
    
    # Final output - simplified to single output for now
    main_output = layers.Conv2D(1, 1, activation='sigmoid', name='main_output')(c8)
    
    model = keras.Model(inputs, main_output)
    return model

# Create the enhanced model
model = enhanced_unet()
print("Enhanced U-Net Architecture:")
model.summary(line_length=120)

# ---------------------------------------------------------------------------
# 8.  Enhanced Loss & metrics with Deep Supervision ------------------------
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

def deep_supervision_loss(y_true, y_pred_list, weights=[1.0, 0.5, 0.25]):
    """Combined loss with deep supervision from multiple outputs"""
    total_loss = 0
    for i, (y_pred, weight) in enumerate(zip(y_pred_list, weights)):
        # Resize y_true to match y_pred if needed
        if y_pred.shape[1:3] != y_true.shape[1:3]:
            y_true_resized = tf.image.resize(y_true, y_pred.shape[1:3])
        else:
            y_true_resized = y_true
        
        loss = masked_bce(y_true_resized, y_pred) + masked_dice(y_true_resized, y_pred)
        total_loss += weight * loss
    
    return total_loss

def combined_model_loss(y_true, y_pred):
    """Wrapper for the multi-output model loss"""
    main_out, aux_out_2, aux_out_4 = y_pred[0], y_pred[1], y_pred[2]
    return deep_supervision_loss(y_true, [main_out, aux_out_2, aux_out_4])

def main_output_iou(y_true, y_pred):
    """IoU metric for the main output only"""
    main_output = y_pred[0] if isinstance(y_pred, list) else y_pred
    return masked_iou(y_true, main_output)

# Compile with enhanced loss and metrics for single output
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),  # Lower LR for stability
    loss=total_loss,
    metrics=[masked_iou]
)

# Updated callbacks for single output
callbacks = [
    ReduceLROnPlateau(monitor='val_masked_iou', mode='max', factor=0.7,
                      patience=6, verbose=1, min_lr=1e-7),
    EarlyStopping(monitor='val_masked_iou', mode='max', patience=15,
                  restore_best_weights=True, verbose=1)
]

# Train the enhanced model
model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds,
          callbacks=callbacks, verbose=2)

# ---------------------------------------------------------------------------
# 9.  Enhanced Prediction & saving for Multi-Output Model ------------------
# ---------------------------------------------------------------------------
THRESH = 0.4

def predict_mask(img):
    """Predict binary mask for a single RGB uint8 image using enhanced U-Net."""
    h0, w0, _ = img.shape
    img_f = img.astype(np.float32) / 255.0
    pad_img = pad_to_multiple(img_f)
    
    # Model returns single output
    main_pred = model.predict(pad_img[np.newaxis])[0, ..., 0]
    
    # Crop back to original size
    pr = main_pred[:h0, :w0]
    pr_bin = (pr >= THRESH).astype(np.uint8)
    
    # Enhanced morphological cleaning with multiple kernel sizes
    kernel_small = np.ones((3,3), np.uint8)
    kernel_medium = np.ones((5,5), np.uint8)
    
    # Remove small noise
    pr_clean = cv2.morphologyEx(pr_bin, cv2.MORPH_OPEN, kernel_small, iterations=1)
    # Fill small holes
    pr_clean = cv2.morphologyEx(pr_clean, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
    # Final smoothing
    pr_clean = cv2.morphologyEx(pr_clean, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # resize/crop to target 375×500
    pr_resized = cv2.resize(pr_clean, (TARGET_SIZE[1], TARGET_SIZE[0]),
                            interpolation=cv2.INTER_NEAREST)
    return pr_resized


print("Saving predictions …")
train_preds = np.stack([predict_mask((im*255).astype(np.uint8))
                        for im in images_train], axis=0)
store_predictions(train_preds, "dataset/train", "unet_v5", fnames_train, palette)

test_preds = np.stack([predict_mask((im*255).astype(np.uint8))
                       for im in images_test], axis=0)
store_predictions(test_preds, "dataset/test1", "unet_v5", fnames_test, palette)
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
