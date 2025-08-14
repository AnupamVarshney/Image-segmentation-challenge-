# unet_simple.py - single, reliable pipeline with 2-image overfit and full class-balanced training
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from util import load_dataset, store_predictions, visualize
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# ----------------------------
# Switches
# ----------------------------
OVERFIT_TWO = False            # Step 1: prove learning on 2 samples (no aug, BCE, high LR)
FULL_CLASS_BALANCED = True   # Step 2: full training with weighted BCE + Dice

# ----------------------------
# Config
# ----------------------------
TARGET_SIZE = (375, 500)
TRAIN_SIZE  = (384, 512)    # divisible by 32; same for train/predict
BATCH_TWO   = 2
EPOCHS_TWO  = 300
LR_TWO      = 1e-3

BATCH_FULL  = 6
EPOCHS_FULL = 150
LR_FULL     = 3e-4
THRESH_FULL = 0.30

# ----------------------------
# Data load
# ----------------------------
print("Loading data…")
images_train, scrib_train, gt_train, fnames_train, palette = load_dataset(
    "dataset/train", "images", "scribbles", "ground_truth"
)
images_test, scrib_test, fnames_test = load_dataset(
    "dataset/test1", "images", "scribbles"
)

# Keep originals for prediction/visualization
images_train_orig = images_train.copy()
images_test_orig  = images_test.copy()

# ----------------------------
# Helpers
# ----------------------------

def resize_pair(img, msk):
    img_r = cv2.resize(img, TRAIN_SIZE[::-1], interpolation=cv2.INTER_LINEAR)
    msk_r = cv2.resize(msk, TRAIN_SIZE[::-1], interpolation=cv2.INTER_NEAREST)
    return img_r, msk_r

def make_features(img_u8, scrib_u8):
    # Resize
    img_r = cv2.resize(img_u8, TRAIN_SIZE[::-1], interpolation=cv2.INTER_LINEAR)
    scrib_r = cv2.resize(scrib_u8.astype(np.uint8), TRAIN_SIZE[::-1], interpolation=cv2.INTER_NEAREST)
    # RGB in [0,1]
    rgb = (img_r.astype(np.float32) / 255.0)
    # LAB in [0,1]
    lab = cv2.cvtColor(img_r, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    # coords in [-1,1]
    h, w = TRAIN_SIZE
    xs = np.linspace(-1, 1, w, dtype=np.float32)
    ys = np.linspace(-1, 1, h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.stack([xx, yy], axis=-1)
    # scribble channel: 0/1; unknown(255)->0.5 neutral
    scrib_ch = np.full((h, w, 1), 0.5, dtype=np.float32)
    scrib_ch[scrib_r == 0] = 0.0
    scrib_ch[scrib_r == 1] = 1.0
    # stack features: 3 + 3 + 2 + 1 = 9 channels
    feats = np.concatenate([rgb, lab, coords, scrib_ch], axis=-1)
    return feats

def preprocess_full(images, masks, scribbles):
    Xs, Ys = [], []
    for img, msk, scr in zip(images, masks, scribbles):
        feats = make_features(img, scr)
        _, msk_r = resize_pair(img, msk)
        Xs.append(feats)
        Ys.append(msk_r)
    X = np.array(Xs).astype(np.float32)
    Y = (np.array(Ys).astype(np.float32))[..., None]
    return X, Y


def build_unet():
    inputs = keras.Input(shape=TRAIN_SIZE + (9,)) # Changed input shape to 9
    
    def conv_block(x, f):
        x = layers.Conv2D(f, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(f, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    
    # Encoder
    c1 = conv_block(inputs, 32); p1 = layers.MaxPooling2D(2)(c1)
    c2 = conv_block(p1, 64);   p2 = layers.MaxPooling2D(2)(c2)
    c3 = conv_block(p2, 128);  p3 = layers.MaxPooling2D(2)(c3)
    c4 = conv_block(p3, 256);  p4 = layers.MaxPooling2D(2)(c4)
    
    # Bottleneck
    bn = conv_block(p4, 512)
    
    # Decoder
    u4 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(bn)
    u4 = layers.Concatenate()([u4, c4])
    u4 = conv_block(u4, 256)
    
    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(u4)
    u3 = layers.Concatenate()([u3, c3])
    u3 = conv_block(u3, 128)
    
    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(u3)
    u2 = layers.Concatenate()([u2, c2])
    u2 = conv_block(u2, 64)
    
    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(u2)
    u1 = layers.Concatenate()([u1, c1])
    u1 = conv_block(u1, 32)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid', dtype='float32')(u1)
    return keras.Model(inputs, outputs)


def iou_metric(y_true, y_pred):
    yb = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(y_true * yb)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(yb) - inter
    return inter / (union + 1e-6)


class DebugCallback(tf.keras.callbacks.Callback):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.x, verbose=0)
        mean_pred = float(np.mean(p))
        # quick IoU on debug set
        pb = (p > 0.5).astype(np.float32)
        inter = float(np.sum(pb * self.y))
        union = float(np.sum(self.y) + np.sum(pb) - inter + 1e-6)
        iou = inter / union
        print(f"  Debug: mean(pred)={mean_pred:.4f} | IoU(two)={iou:.4f}")


def predict_image(model, img_orig, thresh=0.5):
    img_r = cv2.resize(img_orig, TRAIN_SIZE[::-1], interpolation=cv2.INTER_LINEAR)
    img_n = img_r.astype(np.float32) / 255.0
    pred  = model.predict(img_n[None], verbose=0)[0, ..., 0]
    prb   = (pred > thresh).astype(np.uint8)
    pr0   = cv2.resize(prb, (img_orig.shape[1], img_orig.shape[0]), interpolation=cv2.INTER_NEAREST)
    prfin = cv2.resize(pr0, TARGET_SIZE[::-1], interpolation=cv2.INTER_NEAREST)
    return prfin

# ----------------------------
# Step 1: Overfit two samples
# ----------------------------
if OVERFIT_TWO:
    print("\n=== STEP 1: 2-image overfit (BCE, no aug) ===")
    # Pick two samples with foreground
    imgs2, gts2 = [], []
    picked = 0
    for img, gt in zip(images_train, gt_train):
        if np.sum(gt == 1) > 1000:  # ensure some foreground
            imgs2.append(img)
            gts2.append(gt)
            picked += 1
            if picked == 2:
                break
    # fallback if not enough
    while picked < 2:
        imgs2.append(images_train[picked])
        gts2.append(gt_train[picked])
        picked += 1

    X2, Y2 = preprocess_full(imgs2, gts2, scrib_train[:2]) # Pass scribbles for overfit
    print(f"Two-sample shapes: X={X2.shape}, Y={Y2.shape}, FG ratio={np.mean(Y2):.3f}")

    model = build_unet()
    from tensorflow.keras.optimizers.legacy import Adam
    model.compile(optimizer=Adam(LR_TWO), loss='binary_crossentropy', metrics=[iou_metric])
    model.summary()

    model.fit(
        X2, Y2,
        batch_size=BATCH_TWO,
        epochs=EPOCHS_TWO,
        verbose=1,
        callbacks=[DebugCallback(X2, Y2)]
    )

    # Quick sanity predictions on those two
    preds2 = []
    for i in range(2):
        pr = (model.predict(X2[i:i+1], verbose=0)[0, ..., 0] > 0.5).astype(np.uint8)
        preds2.append(cv2.resize(pr, TARGET_SIZE[::-1], interpolation=cv2.INTER_NEAREST))
    preds2 = np.array(preds2)
    # Save to disk for inspection
    store_predictions(preds2, "dataset/train", "unet_overfit_two", fnames_train[:2], palette)

# ----------------------------
# Step 2: Full training with class balance
# ----------------------------
if FULL_CLASS_BALANCED:
    print("\n=== STEP 2: Full training (weighted BCE + Dice) ===")
    X_full, Y_full = preprocess_full(images_train, gt_train, scrib_train) # Pass scribbles for full training

    # simple offline augmentation: horizontal flip
    X_flip = X_full[:, :, ::-1, :]
    Y_flip = Y_full[:, :, ::-1, :]
    X_full = np.concatenate([X_full, X_flip], axis=0)
    Y_full = np.concatenate([Y_full, Y_flip], axis=0)

    # class weights
    pos = float(np.sum(Y_full == 1))
    neg = float(np.sum(Y_full == 0))
    pos_w = (neg / max(pos, 1.0))
    print(f"Positive weight = {pos_w:.2f}, FG ratio = {pos/(pos+neg+1e-6):.3f}")

    def weighted_bce_dice(yt, yp):
        # BCE returns same shape as inputs: [B, H, W, 1]
        bce = tf.keras.backend.binary_crossentropy(yt, yp)  # [B,H,W,1]
        # Build weights with matching shape
        w   = yt * pos_w + (1.0 - yt) * 1.0                # [B,H,W,1]
        wbce= tf.reduce_mean(w * bce)
        # Dice on full tensors with channel
        inter = tf.reduce_sum(yt * yp)
        union = tf.reduce_sum(yt) + tf.reduce_sum(yp)
        dice = 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)
        # Focal term to push positives
        pt = yt * yp + (1.0 - yt) * (1.0 - yp)
        focal = tf.reduce_mean((1.0 - pt) ** 2 * bce)
        # Boundary-aware loss using Sobel edges
        sob = tf.image.sobel_edges
        def edge_map(x):
            e = sob(x)  # [B,H,W,1,2]
            gx = e[..., 0]
            gy = e[..., 1]
            return tf.sqrt(gx * gx + gy * gy)
        edge_loss = tf.reduce_mean(tf.abs(edge_map(yt) - edge_map(yp)))
        return wbce + dice + 0.5 * focal + 0.1 * edge_loss

    model_full = build_unet()
    from tensorflow.keras.optimizers.legacy import Adam
    model_full.compile(optimizer=Adam(LR_FULL), loss=weighted_bce_dice, metrics=[iou_metric])
    model_full.summary()

    callbacks = [
        ReduceLROnPlateau(monitor='val_iou_metric', mode='max', factor=0.5, patience=8, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor='val_iou_metric', mode='max', patience=25, restore_best_weights=True, verbose=1)
    ]

    model_full.fit(
        X_full, Y_full,
        batch_size=BATCH_FULL,
        epochs=EPOCHS_FULL,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1
    )

    # Predictions with simple TTA and post-processing
    print("Generating predictions (full)…")
    
    # Optional DenseCRF post-processing
    def apply_dense_crf(img_u8, mask_prob):
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax
        except Exception:
            return mask_prob  # CRF not available
        H, W = mask_prob.shape
        # Prepare unary (2 classes)
        prob2 = np.zeros((2, H, W), dtype=np.float32)
        prob2[1] = mask_prob
        prob2[0] = 1.0 - mask_prob
        U = unary_from_softmax(prob2)
        d = dcrf.DenseCRF2D(W, H, 2)
        d.setUnaryEnergy(U)
        # Pairwise terms
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=50, srgb=13, rgbim=img_u8, compat=5)
        Q = d.inference(5)
        refined = np.array(Q)[1].reshape(H, W)
        return refined
    
    def keep_components_touching_scribbles(mask_bin, scrib_u8):
        num, lbl, stats, _ = cv2.connectedComponentsWithStats(mask_bin.astype(np.uint8), 8)
        fg = (scrib_u8 == 1)
        keep = np.zeros(num, dtype=bool)
        for cid in range(1, num):
            if np.any((lbl == cid) & fg):
                keep[cid] = True
        out = np.zeros_like(mask_bin)
        for cid in range(1, num):
            if keep[cid]:
                out[lbl == cid] = 1
        return out
    
    def remove_small_components(mask_bin, min_area_ratio=0.0005):
        H, W = mask_bin.shape
        minA = int(min_area_ratio * H * W)
        num, lbl, stats, _ = cv2.connectedComponentsWithStats(mask_bin.astype(np.uint8), 8)
        out = np.zeros_like(mask_bin)
        for cid in range(1, num):
            if stats[cid, cv2.CC_STAT_AREA] >= minA:
                out[lbl == cid] = 1
        return out
    
    def predict_tta(img_u8, scrib_u8=None):
        def predict_single_prob(im, scr):
            feats = make_features(im, scr)
            pr  = model_full.predict(feats[None], verbose=0)[0, ..., 0]
            return pr  # probability map in TRAIN_SIZE
        # TTA on probabilities with multi-scale
        scr = scrib_u8 if scrib_u8 is not None else np.full(img_u8.shape[:2], 255, dtype=np.uint8)
        probs = []
        for scale in [0.75, 1.0, 1.25]:
            if scale == 1.0:
                im_s, sc_s = img_u8, scr
            else:
                im_s = cv2.resize(img_u8, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                sc_s = cv2.resize(scr.astype(np.uint8), (im_s.shape[1], im_s.shape[0]), interpolation=cv2.INTER_NEAREST)
            p0 = predict_single_prob(im_s, sc_s)
            ph = predict_single_prob(im_s[:, ::-1], sc_s[:, ::-1])[:, ::-1]
            pv = predict_single_prob(im_s[::-1, :], sc_s[::-1, :])[::-1, :]
            prob_s = (p0 + ph + pv) / 3.0
            # Resize back to TRAIN_SIZE for averaging
            prob_r = cv2.resize(prob_s, TRAIN_SIZE[::-1], interpolation=cv2.INTER_LINEAR)
            probs.append(prob_r)
        prob = np.mean(probs, axis=0)
        
        # Per-image threshold search using scribbles (wider range)
        thr = THRESH_FULL
        if scrib_u8 is not None and np.any((scrib_u8 == 0) | (scrib_u8 == 1)):
            scrib_r = cv2.resize(scrib_u8.astype(np.uint8), TRAIN_SIZE[::-1], interpolation=cv2.INTER_NEAREST)
            labelled = (scrib_r != 255)
            if np.any(labelled):
                s_true = (scrib_r[labelled] == 1).astype(np.uint8)
                best_iou = -1.0
                best_thr = thr
                for t in np.linspace(0.2, 0.7, 11):
                    s_pred = (prob[labelled] >= t).astype(np.uint8)
                    inter = np.sum(s_true & s_pred)
                    union = np.sum(s_true) + np.sum(s_pred) - inter + 1e-6
                    iou = inter / union
                    if iou > best_iou:
                        best_iou = iou
                        best_thr = float(t)
                thr = best_thr
        
        # Optional DenseCRF refinement on probabilities before threshold
        prob_refined = apply_dense_crf(cv2.resize(img_u8, TRAIN_SIZE[::-1], interpolation=cv2.INTER_LINEAR), prob)
        prb = (prob_refined >= thr).astype(np.uint8)
        
        # Morphological cleaning
        k = np.ones((3,3), np.uint8)
        prb = cv2.morphologyEx(prb, cv2.MORPH_OPEN, k, iterations=1)
        prb = cv2.morphologyEx(prb, cv2.MORPH_CLOSE, k, iterations=2)
        # Scribble-aware filtering
        if scrib_u8 is not None and np.any(scrib_u8 == 1):
            scrib_r = cv2.resize(scrib_u8.astype(np.uint8), TRAIN_SIZE[::-1], interpolation=cv2.INTER_NEAREST)
            prb = keep_components_touching_scribbles(prb, scrib_r)
            prb = remove_small_components(prb)
        # Resize back to original then to target size
        pr0 = cv2.resize(prb, (img_u8.shape[1], img_u8.shape[0]), interpolation=cv2.INTER_NEAREST)
        return cv2.resize(pr0, TARGET_SIZE[::-1], interpolation=cv2.INTER_NEAREST)

    train_preds = []
    for i, (img, scr) in enumerate(zip(images_train_orig, scrib_train)):
        if i % 10 == 0:
            print(f"  Train: {i+1}/{len(images_train_orig)}")
        train_preds.append(predict_tta(img, scr))
    train_preds = np.array(train_preds)

    test_preds = []
    for i, (img, scr) in enumerate(zip(images_test_orig, scrib_test)):
        if i % 10 == 0:
            print(f"  Test: {i+1}/{len(images_test_orig)}")
        test_preds.append(predict_tta(img, scr))
    test_preds = np.array(test_preds)

    print("Saving predictions…")
    store_predictions(train_preds, "dataset/train", "unet_balanced", fnames_train, palette)
    store_predictions(test_preds,  "dataset/test1",  "unet_balanced", fnames_test,  palette)

print("\nRun complete. Toggle OVERFIT_TWO / FULL_CLASS_BALANCED at top to control modes.")

