import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load CIFAR-100 dataset
cifar100 = tf.keras.datasets.cifar100
(X_train, Y_train), (X_test, Y_test) = cifar100.load_data()

# Normalise images
X_train, X_test = X_train / 255.0, X_test / 255.0

# Train-validation split
x_train, x_val, y_train, y_val = train_test_split(
    X_train, Y_train, test_size=0.2
)
y_train = to_categorical(y_train, num_classes=100)
y_val = to_categorical(y_val, num_classes=100)
y_test = to_categorical(Y_test, num_classes=100)

# Image Data Augmentation (Lightweight)
train_datagen = ImageDataGenerator(
    rotation_range=5,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

train_datagen.fit(x_train)

# Load Pretrained MobileNetV2 (96 x 96)
base_model = MobileNetV2(
    input_shape=(96, 96, 3), include_top=False, weights="imagenet"
)

# Freeze all layers except the last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Build Model
model = tf.keras.Sequential(
    [
        tf.keras.layers.UpSampling2D(
            size=(3, 3)
        ),  # Resize CIFAR-100 images (32x32 → 96x96)
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(100, activation="softmax"),
    ]
)

# Compile Model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

# Learning Rate Reduction
lr_reduction = ReduceLROnPlateau(
    monitor="val_accuracy", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)

# Train Model
history = model.fit(
    train_datagen.flow(x_train, y_train, batch_size=64),
    validation_data=(x_val, y_val),
    epochs=20,  # Start with 20 epochs, can increase if needed
    verbose=1,
    callbacks=[lr_reduction],
)

# Plot Loss
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot Accuracy
plt.figure(figsize=(8, 4))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
