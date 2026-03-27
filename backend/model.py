import config
import loading
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    config.DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    config.DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Detected {num_classes} classes.")

#Perfomance optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(20000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# DATA AUGMENTATION
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=config.IMG_SIZE + (3,)
)
base_model.trainable = False
inputs = keras.Input(shape=config.IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(loading.num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining model...")
history = model.fit(
    train_ds,
    validation_data=loading.val_ds,
    epochs=config.EPOCHS
)

#FINE-TUNE
print("\nFine-tuning last 50 layers...")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_ds,
    validation_data=loading.val_ds,
    epochs=10
)

model.save("food_cnn_model.keras")