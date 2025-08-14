import os
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from util import load_dataset, store_predictions, visualize

# ------------------------------------------------------------------------------------------------
# 0) GPU & Reproducibility
# ------------------------------------------------------------------------------------------------
physical_gpus = tf.config.experimental.list_physical_devices('GPU')
if physical_gpus:
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------------------------------------------
# 1) Configuration
# ------------------------------------------------------------------------------------------------
TARGET_SIZE = (375, 500)
PATCH       = 256
BATCH       = 12          # Increased for better gradient estimates
EPOCHS      = 150         # More epochs for better convergence
THRESH      = 0.5         # Standard threshold
L2_WD       = 1e-5        # Reduced for more capacity

# ------------------------------------------------------------------------------------------------
# 2) Load dataset
# ------------------------------------------------------------------------------------------------
images_train, scrib_train, gt_train, fnames_train, palette = load_dataset(
    "dataset/train", "images", "scribbles", "ground_truth"
)
images_test, scrib_test, fnames_test = load_dataset(
    "dataset/test1", "images", "scribbles"
)

images_train_f = images_train.astype(np.float32) / 255.0
images_test_f  = images_test.astype(np.float32)  / 255.0

# Keep originals for inference & viz
images_train_orig = images_train_f.copy()
images_test_orig  = images_test.copy()
scrib_train_orig  = scrib_train.copy()
scrib_test_orig   = scrib_test.copy()
gt_train_orig     = gt_train.copy()

# ------------------------------------------------------------------------------------------------
# 3) Feature engineering & padding
# ------------------------------------------------------------------------------------------------
def pad_to_multiple(arr, m=8, value=0):
    h, w = arr.shape[:2]
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    pad_width = [(0, ph), (0, pw)] + [(0, 0)] * (arr.ndim - 2)
    return np.pad(arr, pad_width, mode='constant', constant_values=value)

def preprocess_features(im01):
    im_u8 = (im01 * 255).astype(np.uint8)
    lab   = cv2.cvtColor(im_u8, cv2.COLOR_RGB2LAB)
    feat  = np.concatenate([im_u8, lab], axis=-1).astype(np.float32) / 255.0
    h, w  = feat.shape[:2]
    xs    = np.linspace(-1, 1, w, dtype=np.float32)
    ys    = np.linspace(-1, 1, h, dtype=np.float32)
    xx, yy= np.meshgrid(xs, ys)
    return np.concatenate([feat, xx[..., None], yy[..., None]], axis=-1)

# Build feature & mask arrays
# --------------------------------
# Original 8-channel features
features = np.array([preprocess_features(im) for im in images_train_f])
# Scribble input channel (0=bg,1=fg,255=ignore)
scrib_input = np.stack([scrib[..., None].astype(np.float32) for scrib in scrib_train], axis=0)
# Normalize scribbles to [0,1]
scrib_input = scrib_input / 255.0
# Concatenate as 9th channel
features_train = [np.concatenate([f, scrib], axis=-1) for f, scrib in zip(features, scrib_input)]
# Stack and pad
features_train = np.array([pad_to_multiple(f, 8) for f in features_train])
mask_train     = np.stack([scrib[..., None].astype(np.float32) for scrib in scrib_train], axis=0)
mask_train     = np.array([pad_to_multiple(m, 8) for m in mask_train])

# Build test features similarly
features_test = np.array([preprocess_features(im) for im in images_test_f])
scrib_test_f  = np.stack([scrib[..., None].astype(np.float32) for scrib in scrib_test], axis=0) / 255.0
features_test = np.array([pad_to_multiple(np.concatenate([f, s], axis=-1), 8)
                           for f, s in zip(features_test, scrib_test_f)])

# ------------------------------------------------------------------------------------------------
# 4) Data pipeline: patch sampling & augmentation
# ------------------------------------------------------------------------------------------------
def np_random_patch(img, msk, max_tries=100, min_labeled=50, min_fg=0.02, max_fg=0.98):
    """Improved patch sampling with better balance"""
    h, w = img.shape[:2]
    th, tw = min(PATCH, h), min(PATCH, w)
    
    for _ in range(max_tries):
        top  = np.random.randint(0, h - th + 1)
        left = np.random.randint(0, w - tw + 1)
        pi   = img[top:top+th, left:left+tw]
        pm   = msk[top:top+th, left:left+tw]
        
        # Check for labeled pixels
        valid = (pm != 255).astype(np.float32)
        if valid.sum() < min_labeled:
            continue
            
        # Check foreground/background balance
        fg_ratio = (pm * valid).sum() / (valid.sum() + 1e-6)
        if min_fg <= fg_ratio <= max_fg:
            return pi, pm
    
    # Fallback: return any patch with enough labeled pixels
    for _ in range(20):
        top  = np.random.randint(0, h - th + 1)
        left = np.random.randint(0, w - tw + 1)
        pi   = img[top:top+th, left:left+tw]
        pm   = msk[top:top+th, left:left+tw]
        valid = (pm != 255).astype(np.float32)
        if valid.sum() >= min_labeled:
            return pi, pm
    
    return pi, pm

def tf_random_patch(img, msk):
    img, msk = tf.numpy_function(np_random_patch, [img, msk], [tf.float32, tf.float32])
    img.set_shape([None, None, 9]); msk.set_shape([None, None, 1])
    img = tf.image.resize_with_pad(img, PATCH, PATCH, antialias=True)
    msk = tf.image.resize_with_pad(msk, PATCH, PATCH, method='nearest')
    img.set_shape([PATCH, PATCH, 9]); msk.set_shape([PATCH, PATCH, 1])
    return img, msk

def augment(img, msk):
    """Enhanced augmentation with more variety"""
    # Geometric augmentations
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img); msk = tf.image.flip_left_right(msk)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img); msk = tf.image.flip_up_down(msk)
    
    # Color augmentations (only on RGB channels)
    rgb = img[..., :3]; lab = img[..., 3:6]; coords = img[..., 6:8]; scrib = img[..., 8:9]
    
    # More aggressive color augmentations for better generalization
    rgb = tf.image.random_brightness(rgb, 0.15)
    rgb = tf.image.random_contrast(rgb, 0.8, 1.2)
    rgb = tf.image.random_saturation(rgb, 0.8, 1.2)
    rgb = tf.image.random_hue(rgb, 0.05)
    
    # Ensure RGB values stay in [0,1]
    rgb = tf.clip_by_value(rgb, 0.0, 1.0)
    
    return tf.concat([rgb, lab, coords, scrib], axis=-1), msk

# Assemble tf.data with better shuffling
N = len(features_train); vs = int(0.15 * N)  # Back to 15% validation
ds = tf.data.Dataset.from_tensor_slices((features_train, mask_train)) \
    .shuffle(N, seed=SEED, reshuffle_each_iteration=True)
val_ds   = ds.take(vs).map(tf_random_patch, tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)
train_ds = ds.skip(vs).map(tf_random_patch, tf.data.AUTOTUNE).map(augment, tf.data.AUTOTUNE) \
    .batch(BATCH).prefetch(tf.data.AUTOTUNE)

# ------------------------------------------------------------------------------------------------
# 5) Model: Enhanced AttentionResUNet with Deep Supervision
# ------------------------------------------------------------------------------------------------
KREG = regularizers.l2(L2_WD) if L2_WD > 0 else None

def AttentionResUNetDS(shape=(None, None, 9)):  # Updated to 9 channels
    inp = keras.Input(shape=shape)
    # Attention gate
    def attention_gate(g, x):
        f = int(g.shape[-1])
        gconv = layers.Conv2D(f,1,padding='same',kernel_regularizer=KREG)(g)
        gconv = layers.BatchNormalization()(gconv)
        xconv = layers.Conv2D(f,1,padding='same',kernel_regularizer=KREG)(x)
        xconv = layers.BatchNormalization()(xconv)
        add = layers.Add()([gconv, xconv])
        relu = layers.Activation('relu')(add)
        psi  = layers.Conv2D(1,1,padding='same',activation='sigmoid',kernel_regularizer=KREG)(relu)
        return layers.Multiply()([x,psi])
    
    # Residual block with more capacity
    def res_conv_block(x, filters, is_first=False, dropout=None):
        shortcut = x
        if is_first:
            x = layers.Conv2D(filters,3,padding='same',kernel_regularizer=KREG)(x)
        else:
            x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
            x = layers.Conv2D(filters,3,padding='same',kernel_regularizer=KREG)(x)
        x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters,3,padding='same',kernel_regularizer=KREG)(x)
        if dropout: x = layers.Dropout(dropout)(x)
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters,1,padding='same',kernel_regularizer=KREG)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        return layers.Add()([x, shortcut])
    
    # Decoder block
    def deconv_block(x, skip, filters):
        x = layers.Conv2DTranspose(filters,2,strides=2,padding='same',kernel_regularizer=KREG)(x)
        x = layers.Concatenate()([x, attention_gate(x, skip)])
        return res_conv_block(x, filters)
    
    # Encoder with more capacity
    c1 = res_conv_block(inp,64,is_first=True); p1 = layers.MaxPooling2D()(c1)  # Increased from 32
    c2 = res_conv_block(p1,128); p2 = layers.MaxPooling2D()(c2)  # Increased from 64
    c3 = res_conv_block(p2,256);p3 = layers.MaxPooling2D()(c3)  # Increased from 128
    c4 = res_conv_block(p3,512,dropout=0.3)  # Increased from 256
    
    # Decoder
    u3 = deconv_block(c4,c3,256); u2 = deconv_block(u3,c2,128); u1 = deconv_block(u2,c1,64)
    
    # Outputs
    out3 = layers.Conv2D(1,1,activation='sigmoid',name='out3')(u3)
    out2 = layers.Conv2D(1,1,activation='sigmoid',name='out2')(u2)
    out1 = layers.Conv2D(1,1,activation='sigmoid',name='out1')(u1)
    return keras.Model(inputs=inp, outputs=[out1,out2,out3], name='AttentionResUNetDS')

model = AttentionResUNetDS()
model.summary(line_length=120)

# ------------------------------------------------------------------------------------------------
# 6) Losses & Metrics
# ------------------------------------------------------------------------------------------------
def masked_bce(y_true,y_pred):
    # Resize ground truth to match prediction size
    y_true_resized = tf.image.resize(y_true, tf.shape(y_pred)[1:3], method='nearest')
    mask = tf.not_equal(y_true_resized,255.0)
    yt = tf.boolean_mask(y_true_resized,mask); yp = tf.boolean_mask(y_pred,mask)
    return tf.reduce_mean(keras.backend.binary_crossentropy(yt,yp))

def masked_focal_loss(y_true,y_pred,alpha=0.25,gamma=2.0):
    """Focal loss for handling class imbalance"""
    # Resize ground truth to match prediction size
    y_true_resized = tf.image.resize(y_true, tf.shape(y_pred)[1:3], method='nearest')
    mask = tf.not_equal(y_true_resized,255.0)
    yt = tf.boolean_mask(y_true_resized,mask); yp = tf.boolean_mask(y_pred,mask)
    
    # Focal loss calculation
    pt = yt * yp + (1 - yt) * (1 - yp)
    focal_weight = alpha * tf.pow(1 - pt, gamma)
    bce = keras.backend.binary_crossentropy(yt, yp)
    return tf.reduce_mean(focal_weight * bce)

def masked_tversky(y_true,y_pred,alpha=0.7,beta=0.3,eps=1e-6):
    # Resize ground truth to match prediction size
    y_true_resized = tf.image.resize(y_true, tf.shape(y_pred)[1:3], method='nearest')
    mask = tf.not_equal(y_true_resized,255.0)
    yt = tf.cast(tf.boolean_mask(y_true_resized,mask),tf.float32)
    yp = tf.cast(tf.boolean_mask(y_pred,mask),tf.float32)
    TP = tf.reduce_sum(yt*yp); FP = tf.reduce_sum((1-yt)*yp)
    FN = tf.reduce_sum(yt*(1-yp))
    return 1 - (TP+eps)/(TP+alpha*FP+beta*FN+eps)

def masked_dice_loss(y_true,y_pred,eps=1e-6):
    """Dice loss for better boundary learning"""
    # Resize ground truth to match prediction size
    y_true_resized = tf.image.resize(y_true, tf.shape(y_pred)[1:3], method='nearest')
    mask = tf.not_equal(y_true_resized,255.0)
    yt = tf.cast(tf.boolean_mask(y_true_resized,mask),tf.float32)
    yp = tf.cast(tf.boolean_mask(y_pred,mask),tf.float32)
    intersection = tf.reduce_sum(yt * yp)
    union = tf.reduce_sum(yt) + tf.reduce_sum(yp)
    return 1 - (2 * intersection + eps) / (union + eps)

def masked_miou(y_true,y_pred):
    # Resize ground truth to match prediction size
    y_true_resized = tf.image.resize(y_true, tf.shape(y_pred)[1:3], method='nearest')
    mask = tf.not_equal(y_true_resized,255.0)
    yt = tf.cast(tf.boolean_mask(y_true_resized,mask),tf.float32)
    yp = tf.cast(tf.boolean_mask(y_pred,mask)>0.5,tf.float32)
    inter = tf.reduce_sum(yt*yp)
    union= tf.reduce_sum(yt)+tf.reduce_sum(yp)-inter
    return inter/(union+1e-6)

# ------------------------------------------------------------------------------------------------
# 7) Compile & Callbacks
# ------------------------------------------------------------------------------------------------
losses = {
    'out1': lambda yt, yp: 0.3*masked_focal_loss(yt,yp) + 0.4*masked_tversky(yt,yp) + 0.3*masked_dice_loss(yt,yp),
    'out2': lambda yt, yp: 0.3*masked_focal_loss(yt,yp) + 0.4*masked_tversky(yt,yp) + 0.3*masked_dice_loss(yt,yp),
    'out3': lambda yt, yp: 0.3*masked_focal_loss(yt,yp) + 0.4*masked_tversky(yt,yp) + 0.3*masked_dice_loss(yt,yp)
}
loss_weights = {'out1':0.7,'out2':0.2,'out3':0.1}  # More weight on main output

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),  # Back to higher learning rate
    loss=losses,
    loss_weights=loss_weights,
    metrics={'out1': masked_miou}
)

cbs = [
    ReduceLROnPlateau(
        monitor='val_out1_masked_miou',
        mode='max',
        factor=0.5,  # Less aggressive reduction
        patience=10,  # More patience
        verbose=1,
        min_lr=1e-6
    ),
    EarlyStopping(
        monitor='val_out1_masked_miou',
        mode='max',
        patience=30,  # More patience
        restore_best_weights=True,
        verbose=1
    )
]

# ------------------------------------------------------------------------------------------------
# 8) Train
# ------------------------------------------------------------------------------------------------
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cbs,
    verbose=2
)

# ------------------------------------------------------------------------------------------------
# 9) Post-processing helpers
# ------------------------------------------------------------------------------------------------
def keep_components_touching_scribbles(pr,scrib):
    """Keep only connected components that touch foreground scribbles"""
    num,lbl,stats,_ = cv2.connectedComponentsWithStats(pr.astype(np.uint8),8)
    fg = (scrib==1); keep = np.zeros(num,bool)
    for cid in range(1,num):
        if np.any((lbl==cid)&fg): keep[cid]=True
    out=np.zeros_like(pr)
    for cid in range(1,num):
        if keep[cid]: out[lbl==cid]=1
    return out

def remove_small_components(pr,min_area_ratio=0.001):  # Increased minimum area
    """Remove small connected components"""
    H,W=pr.shape; minA=int(min_area_ratio*H*W)
    num,lbl,stats,_=cv2.connectedComponentsWithStats(pr.astype(np.uint8),8)
    out=np.zeros_like(pr)
    for cid in range(1,num):
        if stats[cid,cv2.CC_STAT_AREA]>=minA: out[lbl==cid]=1
    return out

def adaptive_threshold(prob_map, scrib):
    """Adaptive thresholding based on scribble information"""
    # If we have foreground scribbles, use adaptive thresholding on those regions
    if np.any(scrib == 1):
        fg_mask = (scrib == 1)
        fg_probs = prob_map[fg_mask]
        if len(fg_probs) > 0:
            # Convert to uint8 for OpenCV (scale 0-1 to 0-255)
            fg_probs_uint8 = (fg_probs * 255).astype(np.uint8)
            try:
                # Use Otsu thresholding on foreground regions
                fg_thresh = cv2.threshold(fg_probs_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
                # Convert back to 0-1 range
                return fg_thresh / 255.0
            except cv2.error:
                # Fallback if Otsu fails
                return np.mean(fg_probs)
    # Fallback to standard threshold
    return THRESH

def enhanced_post_process(pr, scrib):
    """Enhanced post-processing pipeline"""
    # Multi-scale morphological cleaning
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    
    # Opening to remove noise
    pr = cv2.morphologyEx(pr, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Closing to fill holes
    pr = cv2.morphologyEx(pr, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    
    # Keep components touching scribbles
    if np.any(scrib == 1):
        pr = keep_components_touching_scribbles(pr, scrib)
    
    # Remove small components
    pr = remove_small_components(pr)
    
    # Final smoothing
    pr = cv2.morphologyEx(pr, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
    
    return pr

# ------------------------------------------------------------------------------------------------
# 10) TTA inference + predict_mask
# ------------------------------------------------------------------------------------------------
def predict_mask(img_u8,scrib_u8=None):
    def single(im):
        feat=preprocess_features(im.astype(np.float32)/255.0)
        # Add scribble channel if provided, otherwise use zeros
        if scrib_u8 is not None:
            scrib_channel = (scrib_u8.astype(np.float32) / 255.0)[..., None]
        else:
            scrib_channel = np.zeros((feat.shape[0], feat.shape[1], 1), dtype=np.float32)
        feat_with_scrib = np.concatenate([feat, scrib_channel], axis=-1)
        pad=pad_to_multiple(feat_with_scrib,8); h0,w0=feat.shape[:2]
        out1,_,_=model.predict(pad[None],verbose=0)
        prob=out1[0,:h0,:w0,0]
        
        # Adaptive thresholding
        thresh = adaptive_threshold(prob, scrib_u8) if scrib_u8 is not None else THRESH
        pr=(prob>=thresh).astype(np.uint8)
        
        # Enhanced post-processing
        pr = enhanced_post_process(pr, scrib_u8)
        
        return pr
    
    # Test Time Augmentation
    vs=[single(img_u8)]
    
    # Horizontal flip
    hfl=img_u8[:,::-1]
    hfl_scrib = scrib_u8[:,::-1] if scrib_u8 is not None else None
    vs.append(single(hfl)[:,::-1])
    
    # Vertical flip
    vfl=img_u8[::-1,:]
    vfl_scrib = scrib_u8[::-1,:] if scrib_u8 is not None else None
    vs.append(single(vfl)[::-1,:])
    
    # Average predictions
    avg=sum(vs)/3.0
    pr_tta=(avg>=0.5).astype(np.uint8)
    
    # Final post-processing
    if scrib_u8 is not None:
        pr_tta = enhanced_post_process(pr_tta, scrib_u8)
    
    return cv2.resize(pr_tta,(TARGET_SIZE[1],TARGET_SIZE[0]),interpolation=cv2.INTER_NEAREST)

# ------------------------------------------------------------------------------------------------
# 11) Save predictions & visualize
# ------------------------------------------------------------------------------------------------
train_preds=np.stack([predict_mask((im*255).astype(np.uint8),sc)
    for im,sc in zip(images_train_orig,scrib_train_orig)],axis=0)
store_predictions(train_preds,"dataset/train","unet_ds",fnames_train,palette)

test_preds=np.stack([predict_mask(im.astype(np.uint8),sc)
    for im,sc in zip(images_test_orig,scrib_test_orig)],axis=0)
store_predictions(test_preds,"dataset/test1","unet_ds",fnames_test,palette)

idx=np.random.randint(len(images_train))
visualize(
    image=(images_train_orig[idx]*255).astype(np.uint8),
    scribbles=scrib_train_orig[idx],
    ground_truth=gt_train_orig[idx],
    prediction=train_preds[idx],
    pred_title="UNet DS + Boundary Prediction"
)
