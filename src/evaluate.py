
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

MODEL_PATH = "models/fish_quality.keras"
TEST_DIR = Path("data/split/test")

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

model = tf.keras.models.load_model(MODEL_PATH)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224, 224),
    batch_size=16,
    shuffle=False,
    label_mode="binary",
    class_names=["fresh", "spoiled"],
)

y_true = np.concatenate([
    labels.numpy().ravel() for _, labels in test_ds
]).astype(int)

predictions = model.predict(test_ds, verbose=1).ravel()
y_pred = (predictions >= 0.5).astype(int)

print("\nClass order: Fresh = 0, Spoiled = 1")
print(f"Test accuracy: {accuracy_score(y_true, y_pred):.4f}")
print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Fresh", "Spoiled"],
    zero_division=0,
))
print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
