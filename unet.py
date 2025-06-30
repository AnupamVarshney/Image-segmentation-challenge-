import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from util import load_dataset, store_predictions, visualize
from skimage.segmentation import random_walker
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import sys

# ----------- Configuration -----------
TARGET_SIZE = (375, 500)  # (height, width) for final output masks

# ----------- Load Data -----------
images_train, scrib_train, gt_train, fnames_train, palette = load_dataset(
    "dataset/train", "images", "scribbles", "ground_truth"
)
images_test, _, fnames_test = load_dataset(
    "dataset/test1", "images", "scribbles"
)

# ----------- 1. Generate Pseudo-Labels with Random Walker (on original images) -----------
def expand_scribble(img, scribble):
    # Ensure scribble is 2D
    if scribble.ndim == 3:
        scribble = scribble[..., 0]
    
    # Reverting to grayscale since the installed scikit-image is old.
    img_float = img.astype(np.float32) / 255.0
    img_gray = (img_float[..., 0] * 0.299 + img_float[..., 1] * 0.587 + img_float[..., 2] * 0.114)

    # Remap scribble values for random_walker:
    # Unlabeled (255) -> 0
    # Background (0)   -> 1
    # Foreground (1)   -> 2
    seeds = np.zeros_like(scribble, dtype=np.int32)
    seeds[scribble == 0] = 1  # Background seed
    seeds[scribble == 1] = 2  # Foreground seed
    # Pixels that were 255 remain 0, which is what random_walker expects for "unlabeled".
    
    # Using grayscale and a more moderate beta value as a hyperparameter refinement.
    mask = random_walker(img_gray, seeds, beta=500, mode='bf')

    # The output mask will have values 1 (from background seed) and 2 (from foreground seed).
    # We want a binary mask where foreground is 1 and background is 0.
    return (mask == 2).astype(np.uint8)

print("Generating pseudo-labels with Random Walker...")
pseudo_masks_train = np.stack([
    expand_scribble(img, scr)
    for img, scr in zip(images_train, scrib_train)
], axis=0)
print("Pseudo-label generation complete.")

# ----------- 2. Pre-processing and Padding -----------
images_train_processed = images_train.astype(np.float32) / 255.0

def pad_to_multiple(img, multiple=8):
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0: return img
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)) if img.ndim == 3 else ((0, pad_h), (0, pad_w)), mode='constant')

images_train_padded = np.array([pad_to_multiple(img) for img in images_train_processed])
# Now pad the DENSE pseudo-masks
y_train_padded = np.array([pad_to_multiple(mask) for mask in pseudo_masks_train])
y_train_padded = np.expand_dims(y_train_padded, -1).astype(np.float32)

# ----------- 1. U-Net with BatchNorm and Refactored Blocks -----------
def conv_block(x, filters, drop=0.3):
    x = layers.Conv2D(filters, 3, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return layers.Dropout(drop)(x)

def unet(input_shape):
    inputs = keras.Input(shape=input_shape)
    c1 = conv_block(inputs, 16, drop=0.1); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 32, drop=0.1); p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 64, drop=0.2); p3 = layers.MaxPooling2D()(c3)
    b  = conv_block(p3, 128, drop=0.3)
    u3 = layers.UpSampling2D()(b); u3 = layers.concatenate([u3, c3]); c4 = conv_block(u3, 64, drop=0.2)
    u2 = layers.UpSampling2D()(c4); u2 = layers.concatenate([u2, c2]); c5 = conv_block(u2, 32, drop=0.1)
    u1 = layers.UpSampling2D()(c5); u1 = layers.concatenate([u1, c1]); c6 = conv_block(u1, 16, drop=0.1)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c6)
    return keras.Model(inputs, outputs)

# ----------- 2. Class-Balanced Loss Function -----------
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_t = tf.cast(tf.reshape(y_true, [-1]), tf.float32); y_p = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_t * y_p)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_t) + tf.reduce_sum(y_p) + smooth)

def masked_weighted_bce(y_true, y_pred):
    y_t = tf.cast(tf.reshape(y_true, [-1]), tf.float32); y_p = tf.reshape(y_pred, [-1])
    pos = tf.reduce_sum(y_t); neg = tf.reduce_sum(1.0 - y_t)
    alpha = neg / (pos + neg + 1e-6); beta = pos / (pos + neg + 1e-6)
    weights = y_t * alpha + (1 - y_t) * beta
    bce = tf.keras.losses.binary_crossentropy(y_t, y_p)
    return tf.reduce_mean(bce * weights)

def combined_loss(y_true, y_pred):
    return masked_weighted_bce(y_true, y_pred) + dice_loss(y_true, y_pred)

# ----------- Data Augmentation and Pipeline -----------
def augment(image, mask):
    # Basic Flips
    if tf.random.uniform(()) > 0.5: image = tf.image.flip_left_right(image); mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5: image = tf.image.flip_up_down(image); mask = tf.image.flip_up_down(mask)
    
    # Color Augmentation
    if tf.random.uniform(()) > 0.5: image = tf.image.random_brightness(image, max_delta=0.1)
    if tf.random.uniform(()) > 0.5: image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
    # Note: Rotation is more complex and can change image shape. For simplicity, we'll stick to color/flips for now.
    
    return image, mask

dataset_size = len(images_train_padded); val_size = int(0.2 * dataset_size); train_size = dataset_size - val_size
full_dataset = tf.data.Dataset.from_tensor_slices((images_train_padded, y_train_padded)).shuffle(buffer_size=dataset_size)
train_ds = full_dataset.take(train_size).map(augment, num_parallel_calls=tf.data.AUTOTUNE).batch(2).prefetch(tf.data.AUTOTUNE)
val_ds = full_dataset.skip(train_size).batch(2).prefetch(tf.data.AUTOTUNE)

# ----------- 3. Model Setup with LR Scheduling -----------
input_shape = images_train_padded.shape[1:]; model = unet(input_shape)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=combined_loss, metrics=['accuracy'])

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

model.fit(train_ds, epochs=100, validation_data=val_ds, callbacks=callbacks, verbose=1)

# ----------- Prediction and Saving (as before) -----------
def predict_and_save(images, fnames, out_folder, palette):
    preds = []
    for img in images:
        orig_h, orig_w, _ = img.shape
        img_padded = pad_to_multiple(img.astype(np.float32) / 255.0)
        pred = model.predict(np.expand_dims(img_padded, 0))[0, ..., 0]
        # FIX: Invert the prediction. If model outputs < 0.5, it's foreground (1).
        pred_mask = (pred < 0.5).astype(np.uint8)
        cropped_mask = pred_mask[:orig_h, :orig_w]
        preds.append(cropped_mask)
    preds = np.stack(preds, axis=0)
    store_predictions(preds, out_folder, "unet_predictions_v3", fnames, palette)
    return preds

pred_train = predict_and_save(images_train, fnames_train, "dataset/train", palette)
pred_test = predict_and_save(images_test, fnames_test, "dataset/test1", palette)

# ----------- Visualization -----------
vis_index = np.random.randint(len(images_train))
visualize(images_train[vis_index], scrib_train[vis_index], gt_train[vis_index], pred_train[vis_index])
