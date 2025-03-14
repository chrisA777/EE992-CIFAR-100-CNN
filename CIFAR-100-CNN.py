# -*- coding: utf-8 -*-
"""
EE992: Neural Networks and Deep Learning

Image Classification Project on CIFAR-100 Data Set
Chris Abi-Aad & Sean Doherty

"""

# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
    BatchNormalization,
    Activation,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random

# Load Data Set
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data(
    label_mode="fine"
)

# Show 10 random example images
images = []
for i in range(10):
    n = random.randint(0, X_train.shape[0] - 1)
    images.append(X_train[n])

images = np.hstack(images)
plt.figure(figsize=(15, 5))
plt.imshow(images)
plt.axis("off")
plt.show()

# Process Data
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Transform labels to one hot encoding
y_train = keras.utils.to_categorical(y_train, 100)
y_test = keras.utils.to_categorical(y_test, 100)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=93
)
train_datagen.fit(X_train)


# Define a Simplified Model
def create_model():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding="same", input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation="softmax"))

    # Compile Model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
        metrics=["accuracy"],
    )

    return model


# Train Model
model = create_model()
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=64),
    steps_per_epoch=100,
    epochs=200,
    validation_data=(X_val, y_val),
    verbose=1,
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
