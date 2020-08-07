from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from consts import *
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

from keras.engine import Model
from keras.layers import Dropout, Flatten, Dense
from keras.applications import VGG16

custom_model = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', 
           input_shape=(IMG_SIZE,IMG_SIZE,depth)),
    MaxPooling2D(pool_size=2),
    Conv2D(64, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(128, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Dropout(0.4),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.3),
    Dense(1024, activation='relu'),
    Dense(2, activation='softmax'),
])

input_shape = (IMG_SIZE, IMG_SIZE, depth)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
last = base_model.layers[-1].output
x = Flatten()(last)
x = Dense(512, activation='relu', name='fc1')(x)
x = Dropout(0.3)(x)
x = Dense(2, activation='softmax', name='predictions')(x)
VGG_model = Model(base_model.input, x)
