import tensorflow as tf
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from CNN_model import VGG_model
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

n_max = 20000

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

for layer in (VGG_model.layers)[:15]:
    layer.trainable = False
    print('Layer ' + layer.name + ' frozen.')



VGG_model.compile(
    optimizer=Adam(lr=0.001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

VGG_model.summary()


batch_size = 60

trdata = ImageDataGenerator()
train_data = trdata.flow(x=train_images, y=train_labels, batch_size=batch_size)
tsdata = ImageDataGenerator()
test_data = tsdata.flow(x=test_images, y=test_labels, batch_size=batch_size)

checkpoint = ModelCheckpoint("ieeercnn_vgg16_1.h5", monitor='val_loss', 
                             verbose=1, save_best_only=True, 
                             save_weights_only=False, mode='auto', period=1)

# We train it
VGG_model.fit(train_data,
          validation_data=test_data,
          epochs=N_EPOCHS,
          verbose=2,
          callbacks=[checkpoint],
)

VGG_model.save_weights(weights_dir + 'cnn_fine_' + str(CLASS_ID) + '.h5')

