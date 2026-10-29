
from pathlib import Path
import tensorflow as tf

# Dataset folder
DATA_DIR = Path("data/split")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

print("Dataset folder:", DATA_DIR.resolve())
print("Class folders:", [p.name for p in DATA_DIR.iterdir() if p.is_dir()])


# Load training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    seed=SEED,
)

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    seed=SEED,
)


train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# Transfer learning with MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
)

Path("models").mkdir(exist_ok=True)
model.save("models/fish_quality.keras")

print("Training complete!")
print("Model saved to models/fish_quality.keras")
