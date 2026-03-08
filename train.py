import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ---------------- CONFIG ---------------- #

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
EPOCHS = 12

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.txt")

SEED = 42


# ---------------- DATA ---------------- #

def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_ds.class_names

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


def compute_class_weights(class_names):
    counts = {}
    for class_name in class_names:
        folder = Path(TRAIN_DIR) / class_name
        counts[class_name] = len([p for p in folder.iterdir() if p.is_file()])

    total = sum(counts.values())
    num_classes = len(class_names)

    class_weights = {}
    for idx, class_name in enumerate(class_names):
        count = counts[class_name]
        class_weights[idx] = total / (num_classes * count) if count > 0 else 1.0

    print("\nTraining image counts:")
    for name, count in counts.items():
        print(f"  {name}: {count}")

    print("\nClass weights:")
    for idx, name in enumerate(class_names):
        print(f"  {name}: {class_weights[idx]:.4f}")

    return class_weights


# ---------------- MODEL ---------------- #

def build_model(num_classes):
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
        layers.RandomTranslation(0.08, 0.08),
    ], name="augmentation")

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="vision_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ---------------- TRAIN ---------------- #

def main():
    print("Loading datasets...")
    train_ds, val_ds, class_names = load_datasets()
    num_classes = len(class_names)

    print("\nClasses found:")
    print(class_names)

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(CLASS_NAMES_PATH, "w") as f:
        for name in class_names:
            f.write(name + "\n")

    class_weights = compute_class_weights(class_names)
    model = build_model(num_classes)

    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    print("\nStarting training...\n")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights
    )

    model.save(MODEL_PATH)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved class names to {CLASS_NAMES_PATH}")

    val_loss, val_acc = model.evaluate(val_ds, verbose=1)
    print(f"\nFinal validation loss: {val_loss:.4f}")
    print(f"Final validation accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()