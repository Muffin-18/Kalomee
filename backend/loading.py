import config
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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