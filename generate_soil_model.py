import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Dummy training data
num_classes = 5
num_samples = 100
X_dummy = np.random.rand(num_samples, 150, 150, 3)
y_dummy = np.random.randint(0, num_classes, num_samples)
y_dummy = to_categorical(y_dummy, num_classes=num_classes)

# Build CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_dummy, y_dummy, epochs=3)

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/soil_model.h5")

print("✅ Dummy soil_model.h5 generated in /model/")

