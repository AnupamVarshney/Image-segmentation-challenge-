import numpy as np
from util import load_dataset, store_predictions, visualize
from sklearn.ensemble import RandomForestClassifier

# ----------- Load Data -----------
# Training data
images_train, scrib_train, gt_train, fnames_train, palette = load_dataset(
    "dataset/train", "images", "scribbles", "ground_truth"
)

# Test data
images_test, scrib_test, fnames_test = load_dataset(
    "dataset/test1", "images", "scribbles"
)

# ----------- Prepare Training Data -----------
X = []
y = []

for img, scrib in zip(images_train, scrib_train):
    H, W, _ = img.shape
    for i in range(H):
        for j in range(W):
            label = scrib[i, j]
            if label != 255:  # Only use labeled pixels
                X.append([img[i, j, 0], img[i, j, 1], img[i, j, 2], i, j])
                y.append(label)

X = np.array(X)
y = np.array(y)

# ----------- Train Random Forest -----------
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
clf.fit(X, y)

# ----------- Predict Function -----------
def predict_mask(img, clf):
    H, W, _ = img.shape
    X_test = []
    for i in range(H):
        for j in range(W):
            X_test.append([img[i, j, 0], img[i, j, 1], img[i, j, 2], i, j])
    X_test = np.array(X_test)
    y_pred = clf.predict(X_test)
    return y_pred.reshape(H, W)

# ----------- Predict on Training Set -----------
pred_train = []
for img in images_train:
    pred_mask = predict_mask(img, clf)
    pred_train.append(pred_mask)
pred_train = np.stack(pred_train, axis=0)

# ----------- Predict on Test Set -----------
pred_test = []
for img in images_test:
    pred_mask = predict_mask(img, clf)
    pred_test.append(pred_mask)
pred_test = np.stack(pred_test, axis=0)

# ----------- Store Predictions -----------
store_predictions(pred_train, "dataset/train", "rf_predictions", fnames_train, palette)
store_predictions(pred_test, "dataset/test1", "rf_predictions", fnames_test, palette)

# ----------- Evaluate and Visualize -----------


# Visualize a random result
vis_index = np.random.randint(images_train.shape[0])
visualize(
    images_train[vis_index], scrib_train[vis_index],
    gt_train[vis_index], pred_train[vis_index]
)