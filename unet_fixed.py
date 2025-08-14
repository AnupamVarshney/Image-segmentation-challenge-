# unet_fixed.py
# COMPLETELY FIXED UNet implementation addressing all fundamental issues
import os
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from util import load_dataset, store_predictions, visualize
import datetime
import time

# ------------------------------------------------------------------------------------------------
# 0) GPU & Reproducibility
# ------------------------------------------------------------------------------------------------
physical_gpus = tf.config.experimental.list_physical_devices('GPU')
if physical_gpus:
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"🚀 Found {len(physical_gpus)} GPU(s), memory growth enabled")

SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------------------------------------------
# 1) Configuration - FIXED FOR PROPER SEGMENTATION
# ------------------------------------------------------------------------------------------------
TARGET_SIZE = (375, 500)
TRAIN_SIZE = (384, 512)   # Consistent training size
BATCH = 8                 # Reasonable batch for full images
EPOCHS = 80               # Sufficient epochs
THRESH = 0.5              # Standard threshold
L2_WD = 1e-5              # Regularization

print("🔧 CRITICAL FIXES APPLIED:")
print("   ✅ Training on GROUND TRUTH masks (not scribbles)")
print("   ✅ Full image training (no patches)")
print("   ✅ Consistent train/test pipeline")
print("   ✅ Proper loss functions")

# ------------------------------------------------------------------------------------------------
# 2) Load dataset
# ------------------------------------------------------------------------------------------------
print("🔄 Loading dataset...")
images_train, scrib_train, gt_train, fnames_train, palette = load_dataset(
    "dataset/train", "images", "scribbles", "ground_truth"
)
images_test, scrib_test, fnames_test = load_dataset(
    "dataset/test1", "images", "scribbles"
)

images_train_f = images_train.astype(np.float32) / 255.0
images_test_f = images_test.astype(np.float32) / 255.0

# Keep originals
images_train_orig = images_train_f.copy()
images_test_orig = images_test.copy()
scrib_train_orig = scrib_train.copy()
scrib_test_orig = scrib_test.copy()
gt_train_orig = gt_train.copy()

print(f"📊 Training images: {len(images_train)}")
print(f"📊 Test images: {len(images_test)}")

# ------------------------------------------------------------------------------------------------
# 3) Data preprocessing - FULL IMAGE APPROACH
# ------------------------------------------------------------------------------------------------
def preprocess_features(img):
    """Feature engineering for full images"""
    # Resize to consistent training size
    img_resized = cv2.resize(img, TRAIN_SIZE[::-1], interpolation=cv2.INTER_LINEAR)
    img_u8 = (img_resized * 255).astype(np.uint8)
    
    # Add LAB color space
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    feat = np.concatenate([img_u8, lab], axis=-1).astype(np.float32) / 255.0
    
    # Add coordinate channels
    h, w = feat.shape[:2]
    xs = np.linspace(-1, 1, w, dtype=np.float32)
    ys = np.linspace(-1, 1, h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    
    return np.concatenate([feat, xx[..., None], yy[..., None]], axis=-1)

# Process training data
print("🔧 Processing training data with GROUND TRUTH masks...")
features_train = []
masks_train = []

for i, (img, scrib, gt) in enumerate(zip(images_train_f, scrib_train, gt_train)):
    if i % 10 == 0:
        print(f"   Processing {i+1}/{len(images_train)}...")
    
    # Feature engineering
    feat = preprocess_features(img)
    
    # Add scribble guidance channel
    scrib_resized = cv2.resize(scrib.astype(np.float32), TRAIN_SIZE[::-1], interpolation=cv2.INTER_NEAREST)
    scrib_channel = (scrib_resized / 255.0)[..., None]
    feat_with_scrib = np.concatenate([feat, scrib_channel], axis=-1)
    
    # CRITICAL FIX: Use ground truth masks!
    gt_resized = cv2.resize(gt.astype(np.float32), TRAIN_SIZE[::-1], interpolation=cv2.INTER_NEAREST)
    mask = gt_resized[..., None]
    
    features_train.append(feat_with_scrib)
    masks_train.append(mask)

features_train = np.array(features_train)
masks_train = np.array(masks_train)

print(f"✅ Training features shape: {features_train.shape}")
print(f"✅ Training masks shape: {masks_train.shape}")

# Process test data
features_test = []
for img, scrib in zip(images_test_f, scrib_test):
    feat = preprocess_features(img)
    scrib_resized = cv2.resize(scrib.astype(np.float32), TRAIN_SIZE[::-1], interpolation=cv2.INTER_NEAREST)
    scrib_channel = (scrib_resized / 255.0)[..., None]
    feat_with_scrib = np.concatenate([feat, scrib_channel], axis=-1)
    features_test.append(feat_with_scrib)
features_test = np.array(features_test)

# ------------------------------------------------------------------------------------------------
# 4) Data pipeline - FULL IMAGE TRAINING
# ------------------------------------------------------------------------------------------------
def augment_full_image(img, msk):
    """Augmentation for full images"""
    # Geometric augmentations
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        msk = tf.image.flip_left_right(msk)
    
    # Color augmentations (only on RGB channels)
    rgb = img[..., :3]
    lab = img[..., 3:6]
    coords = img[..., 6:8]
    scrib = img[..., 8:9]
    
    # Conservative augmentations
    rgb = tf.image.random_brightness(rgb, 0.1)
    rgb = tf.image.random_contrast(rgb, 0.9, 1.1)
    rgb = tf.image.random_saturation(rgb, 0.9, 1.1)
    rgb = tf.clip_by_value(rgb, 0.0, 1.0)
    
    return tf.concat([rgb, lab, coords, scrib], axis=-1), msk

# Create datasets
N = len(features_train)
vs = int(0.15 * N)

ds = tf.data.Dataset.from_tensor_slices((features_train, masks_train))
ds = ds.shuffle(N, seed=SEED, reshuffle_each_iteration=True)

val_ds = ds.take(vs).batch(BATCH).prefetch(tf.data.AUTOTUNE)
train_ds = ds.skip(vs).map(augment_full_image, tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)

# ------------------------------------------------------------------------------------------------
# 5) Model - Simplified UNet for full images
# ------------------------------------------------------------------------------------------------
def SimpleUNet(input_shape=(384, 512, 9)):
    """Simple but effective UNet for full image segmentation"""
    inputs = keras.Input(shape=input_shape)
    
    # Encoder
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D(2)(c3)
    
    c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(c4)
    c4 = layers.Dropout(0.3)(c4)
    
    # Decoder
    u3 = layers.UpSampling2D(2)(c4)
    u3 = layers.Conv2D(128, 3, activation='relu', padding='same')(u3)
    u3 = layers.Concatenate()([u3, c3])
    u3 = layers.Conv2D(128, 3, activation='relu', padding='same')(u3)
    u3 = layers.Conv2D(128, 3, activation='relu', padding='same')(u3)
    
    u2 = layers.UpSampling2D(2)(u3)
    u2 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
    u2 = layers.Concatenate()([u2, c2])
    u2 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
    u2 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
    
    u1 = layers.UpSampling2D(2)(u2)
    u1 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)
    u1 = layers.Concatenate()([u1, c1])
    u1 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)
    u1 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)
    
    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(u1)
    
    return keras.Model(inputs, outputs, name='SimpleUNet')

model = SimpleUNet()
print("🏗️  Model Architecture:")
model.summary()

# ------------------------------------------------------------------------------------------------
# 6) Loss functions - PROPER SEGMENTATION LOSSES
# ------------------------------------------------------------------------------------------------
def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice loss for segmentation"""
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1 - (2 * intersection + smooth) / (union + smooth)

def combined_loss(y_true, y_pred):
    """Combined BCE + Dice loss"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return tf.reduce_mean(bce) + dice

def iou_metric(y_true, y_pred):
    """IoU metric"""
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_binary)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_binary) - intersection
    return intersection / (union + 1e-6)

# ------------------------------------------------------------------------------------------------
# 7) Compile and train
# ------------------------------------------------------------------------------------------------
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=combined_loss,
    metrics=[iou_metric]
)

callbacks = [
    ReduceLROnPlateau(monitor='val_iou_metric', mode='max', factor=0.5, patience=8, verbose=1),
    EarlyStopping(monitor='val_iou_metric', mode='max', patience=20, restore_best_weights=True, verbose=1)
]

print("🚀 Starting FIXED UNet training...")
print("   ✅ Training on ground truth masks")
print("   ✅ Full image approach")
print("   ✅ Proper loss functions")

start_time = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
training_time = time.time() - start_time

print(f"✅ Training completed in {training_time/60:.1f} minutes!")

# Save model
model.save('unet_fixed.h5')
print("💾 Model saved as 'unet_fixed.h5'")

# ------------------------------------------------------------------------------------------------
# 8) Prediction - CONSISTENT WITH TRAINING
# ------------------------------------------------------------------------------------------------
def predict_mask_fixed(img_u8, scrib_u8):
    """Fixed prediction function consistent with training"""
    # Convert to float
    img_f = img_u8.astype(np.float32) / 255.0
    
    # Same preprocessing as training
    feat = preprocess_features(img_f)
    scrib_resized = cv2.resize(scrib_u8.astype(np.float32), TRAIN_SIZE[::-1], interpolation=cv2.INTER_NEAREST)
    scrib_channel = (scrib_resized / 255.0)[..., None]
    feat_with_scrib = np.concatenate([feat, scrib_channel], axis=-1)
    
    # Predict
    pred = model.predict(feat_with_scrib[None], verbose=0)[0, ..., 0]
    
    # Threshold
    pred_binary = (pred > THRESH).astype(np.uint8)
    
    # Resize back to original size
    h0, w0 = img_u8.shape[:2]
    pred_resized = cv2.resize(pred_binary, (w0, h0), interpolation=cv2.INTER_NEAREST)
    
    # Final resize to target
    pred_final = cv2.resize(pred_resized, TARGET_SIZE[::-1], interpolation=cv2.INTER_NEAREST)
    
    return pred_final

# ------------------------------------------------------------------------------------------------
# 9) Generate predictions
# ------------------------------------------------------------------------------------------------
print("🔮 Generating predictions...")

# Training predictions
train_preds = []
for i, (img, scrib) in enumerate(zip(images_train_orig, scrib_train_orig)):
    if i % 10 == 0:
        print(f"   Train: {i+1}/{len(images_train_orig)}")
    pred = predict_mask_fixed((img * 255).astype(np.uint8), scrib)
    train_preds.append(pred)
train_preds = np.array(train_preds)

# Test predictions
test_preds = []
for i, (img, scrib) in enumerate(zip(images_test_orig, scrib_test_orig)):
    if i % 10 == 0:
        print(f"   Test: {i+1}/{len(images_test_orig)}")
    pred = predict_mask_fixed(img.astype(np.uint8), scrib)
    test_preds.append(pred)
test_preds = np.array(test_preds)

# ------------------------------------------------------------------------------------------------
# 10) Save and visualize
# ------------------------------------------------------------------------------------------------
print("💾 Saving predictions...")
store_predictions(train_preds, "dataset/train", "unet_fixed", fnames_train, palette)
store_predictions(test_preds, "dataset/test1", "unet_fixed", fnames_test, palette)

# Visualize
idx = random.randint(0, len(train_preds)-1)
print(f"🎨 Visualizing example {idx}...")
visualize(
    image=(images_train_orig[idx] * 255).astype(np.uint8),
    scribbles=scrib_train_orig[idx],
    ground_truth=gt_train_orig[idx],
    prediction=train_preds[idx],
    pred_title="FIXED UNet Prediction"
)

print("\n" + "="*60)
print("🎉 FIXED UNET TRAINING COMPLETED!")
print("="*60)
print("✅ All fundamental issues addressed:")
print("   - Training on ground truth masks (not scribbles)")
print("   - Full image training (no patch mismatch)")
print("   - Consistent train/test pipeline")
print("   - Proper segmentation losses")
print("="*60)
