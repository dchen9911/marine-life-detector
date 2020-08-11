import numpy as np
import glob
import os 
import matplotlib.image as mpimg
import tensorflow as tf
import cv2
from keras.utils import to_categorical
from keras import backend as K
from consts import *

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

from CNN_model import model
from skimage import color
from skimage import io

class_name = CLASS_NAMES[CLASS_ID]

# load the training data
train_path = base_path + 'images/cropped_resized/train/'
test_path = base_path + 'images/cropped_resized/test/'

if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)

train_images = []
train_labels = []

test_images = []
test_labels = []

n_other = 0

if CLASS_ID == 0:
    n_max = 100000
else:
    n_max = 1000

# for filepath in glob.glob(train_path + '*.*'):
#     i_label = 5
#     for i in range(5):
#         if CLASS_NAMES[i] in filepath:
#             i_label = i
#             break
#     if i_label == 5:
#         n_other += 1
#     train_labels.append(i_label)
#     train_images.append(mpimg.imread(filepath))


# for filepath in glob.glob(test_path + '*.*'):
#     i_label = 5
#     for i in range(5):
#         if CLASS_NAMES[i] in filepath:
#             i_label = i
#             break
#     if i_label == 5:
#         n_other += 1
#     test_labels.append(i_label)
#     test_images.append(mpimg.imread(filepath))

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

batch_size = 60

trdata = ImageDataGenerator()
train_data = trdata.flow(x=train_images, y=train_labels, batch_size=batch_size)
tsdata = ImageDataGenerator()
test_data = tsdata.flow(x=test_images, y=test_labels, batch_size=batch_size)

model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.fit(
  train_data,
  epochs=N_EPOCHS,
  steps_per_epoch=n_batches,
  validation_data = test_data,
  verbose=2,
)

# model.fit(
#   train_images,
#   to_categorical(train_labels),
#   epochs=20 ,
#   validation_split=0.1,
#   validation_data = (test_images, to_categorical(test_labels)),
#   verbose=2,
# )

# loss, accuracy, f1_score, precision, recall = model.evaluate(test_images, to_categorical(test_labels), verbose=1)

model.save_weights(weights_dir + 'cnn_fine_' + str(CLASS_ID) + '.h5')

print("done")
