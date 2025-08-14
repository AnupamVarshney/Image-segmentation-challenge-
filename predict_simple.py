"""Simple prediction script for the trained Simple UNet model"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from util import load_dataset, store_predictions, visualize

# Load the trained model
print("Loading trained Simple UNet model...")
model = keras.models.load_model('simple_unet_model.h5', compile=False)
print("Model loaded successfully!")

# Load dataset
print("Loading dataset...")
images_train, scrib_train, gt_train, fnames_train, palette = load_dataset(
    "dataset/train", "images", "scribbles", "ground_truth")
images_test, _, fnames_test = load_dataset("dataset/test1", "images", "scribbles")

print(f"Dataset loaded: {len(images_train)} train, {len(images_test)} test images")

# Normalize images
images_train = images_train.astype(np.float32) / 255.0
images_test = images_test.astype(np.float32) / 255.0

def pad_to_multiple(arr, m=8, value=0):
    h, w = arr.shape[:2]
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph == 0 and pw == 0:
        return arr
    pad_spec = ((0, ph), (0, pw)) + (() if arr.ndim == 2 else ((0, 0),))
    return np.pad(arr, pad_spec, mode='constant', constant_values=value)

def predict_mask(img):
    """Predict binary mask for a single RGB uint8 image"""
    h0, w0, _ = img.shape
    img_f = img.astype(np.float32) / 255.0
    pad_img = pad_to_multiple(img_f)
    
    # Model prediction
    pred = model.predict(pad_img[np.newaxis], verbose=0)[0, ..., 0]
    
    # Crop back to original size
    pr = pred[:h0, :w0]
    pr_bin = (pr >= 0.4).astype(np.uint8)
    
    # Morphological cleaning
    kernel = np.ones((3,3), np.uint8)
    pr_clean = cv2.morphologyEx(pr_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    pr_clean = cv2.morphologyEx(pr_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # resize/crop to target 375×500
    pr_resized = cv2.resize(pr_clean, (500, 375), interpolation=cv2.INTER_NEAREST)
    return pr_resized

# Make predictions
print("Making predictions...")
train_preds = np.stack([predict_mask((im*255).astype(np.uint8))
                        for im in images_train], axis=0)
store_predictions(train_preds, "dataset/train", "simple_unet_fixed", fnames_train, palette)

test_preds = np.stack([predict_mask((im*255).astype(np.uint8))
                       for im in images_test], axis=0)
store_predictions(test_preds, "dataset/test1", "simple_unet_fixed", fnames_test, palette)

print("Predictions saved successfully!")

# Show a sample result
idx = np.random.randint(len(images_train))
print(f"Sample prediction for image {idx}:")
visualize((images_train[idx]*255).astype(np.uint8),
          scrib_train[idx],
          gt_train[idx],
          train_preds[idx]) 