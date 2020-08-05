import tensorflow as tf
from keras import callbacks
from keras import optimizers
from keras.engine import Model
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.applications import VGG16
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from consts import *
import glob
import numpy as np
import os 
import matplotlib.image as mpimg
from skimage import color
from skimage import io

class_name = CLASS_NAMES[CLASS_ID]

# load the training data
train_path = base_path + 'images/cropped_resized/train/'
test_path = base_path + 'images/cropped_resized/test/'

train_images = []
train_labels = []

test_images = []
test_labels = []

n_other = 0

n_max = 10000

for filepath in glob.glob(train_path + '*.*'):
    if USE_GREY:
        img = io.imread(filepath, as_gray=USE_GREY)
    else:
        img = mpimg.imread(filepath)

    if class_name in filepath:
        train_labels.append(1)
        train_images.append(img)
    else:
        if n_other > n_max:
            continue
        train_labels.append(0)
        train_images.append(img)
        n_other += 1

for filepath in glob.glob(test_path + '*.*'):
    if USE_GREY:
        img = io.imread(filepath, as_gray=USE_GREY)
    else:
        img = mpimg.imread(filepath)
    
    test_images.append(img)
    if class_name in filepath:
        test_labels.append(1)
    else:
        test_labels.append(0)
        
if USE_GREY:
    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)

print('Positive examples: ' + str(len(train_images) - n_other) + '/' + str(len(train_images)))
train_labels = np.array(train_labels)
train_images = np.array(train_images)
train_images -= 0.5

print(train_images.shape)

test_labels = np.array(test_labels)
test_images = np.array(test_images)
test_images -= 0.5

print(test_images.shape)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1124)])

input_shape = (IMG_SIZE, IMG_SIZE, depth)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

for i in range(len(base_model.layers)): 
    if i == 11:
        break
    layer = base_model.layers[i]
    layer.trainable = False
    print('Layer ' + layer.name + ' frozen.')

last = base_model.layers[-1].output
x = Flatten()(last)
x = Dense(512, activation='relu', name='fc1')(x)
x = Dropout(0.3)(x)
x = Dense(2, activation='softmax', name='predictions')(x)
model = Model(base_model.input, x)
# We compile the model

model.compile(
    optimizer=Adam(lr=0.001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

model.summary()

epochs = 10
batch_size = 60

trdata = ImageDataGenerator()
train_data = trdata.flow(x=train_images, y=train_labels, batch_size=batch_size)
tsdata = ImageDataGenerator()
test_data = tsdata.flow(x=test_images, y=test_labels, batch_size=batch_size)

# We train it
model.fit(train_data,
          validation_data=test_data,
          epochs=epochs,
          verbose=2,
)

model.save_weights(weights_dir + 'cnn_fine_' + str(CLASS_ID) + '.h5')

