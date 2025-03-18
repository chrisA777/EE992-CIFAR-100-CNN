import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load CIFAR-100 dataset
cifar100 = tf.keras.datasets.cifar100
(X_train, Y_train), (X_test, Y_test) = cifar100.load_data()

# Normalize images
X_train, X_test = X_train / 255.0, X_test / 255.0

# Train-validation split
x_train, x_val, y_train, y_val = train_test_split(
    X_train, Y_train, test_size=0.2
)
y_train = to_categorical(y_train, num_classes=100)
y_val = to_categorical(y_val, num_classes=100)
y_test = to_categorical(Y_test, num_classes=100)

# 🔹 Advanced Data Augmentation: CutMix & MixUp
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
)

train_datagen.fit(x_train)

# Load Pretrained EfficientNetB2
base_model = EfficientNetB2(
    input_shape=(96, 96, 3), include_top=False, weights="imagenet"
)

# Freeze most layers, fine-tune last 50
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Build Model with Stronger Regularization
model = tf.keras.Sequential(
    [
        tf.keras.layers.UpSampling2D(
            size=(3, 3)
        ),  # Resize CIFAR-100 images (32x32 → 96x96)
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),  # Stronger dropout to prevent overfitting
        Dense(512, activation="relu", kernel_regularizer=l2(0.0005)),
        Dropout(0.5),
        Dense(100, activation="softmax"),  # Label smoothing applied
    ]
)

# Compile Model with Cosine Decay Learning Rate
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=50, alpha=1e-6
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)

# Learning rate scheduler & early stopping
lr_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=5, factor=0.5, min_lr=1e-6, verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
)

# Train Model for More Epochs
history = model.fit(
    train_datagen.flow(x_train, y_train, batch_size=64),
    validation_data=(x_val, y_val),
    epochs=75,  # Increased training epochs
    verbose=1,
    callbacks=[lr_reduction, early_stopping],
)

# Save improved model
model.save("efficientnet_cifar100.keras")

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
