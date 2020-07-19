from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from consts import IMG_SIZE
import tensorflow as tf

model = Sequential([
    Conv2D(16, kernel_size=3, activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,3)),
    MaxPooling2D(pool_size=2),
    Conv2D(32, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(64, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    # Dropout(0.5),
    Dense(2, activation='softmax'),
])