import numpy as np
import glob
import os 
import matplotlib.image as mpimg
import tensorflow as tf
from keras.utils import to_categorical
from keras import backend as K
from consts import IMG_SIZE, CLASS_NAMES, CLASS_ID, base_path

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

class_name = CLASS_NAMES[CLASS_ID]

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=900)])
# tf.config.experimental.set_memory_growth(gpus[0], True)

from CNN_model import model

# load the training data
train_path = base_path + 'images/cropped_resized/train/'
test_path = base_path + 'images/cropped_resized/test/'

weights_dir = base_path + '/weights/'

if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)

train_images = []
train_labels = []

test_images = []
test_labels = []

n_other = 0

if CLASS_ID == 0:
    n_max = 10000
else:
    n_max = 1000

for filepath in glob.glob(train_path + '*.*'):
    if class_name in filepath:
        train_labels.append(1)
        train_images.append(mpimg.imread(filepath))
    else:
        if n_other > n_max:
            continue
        train_labels.append(0)
        train_images.append(mpimg.imread(filepath))
        n_other += 1

for filepath in glob.glob(test_path + '*.*'):
    test_images.append(mpimg.imread(filepath))
    if class_name in filepath:
        test_labels.append(1)
    else:
        test_labels.append(0)

print('Positive examples: ' + str(len(train_images) - n_other) + '/' + str(len(train_images)))
train_labels = np.array(train_labels)
train_images = np.array(train_images)
train_images -= 0.5

print(train_images.shape)

test_labels = np.array(test_labels)
test_images = np.array(test_images)
test_images -= 0.5

print(test_images.shape)

model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

def generate_data(train_images, train_labels, n_batches):
    i = 0

    batch_size = int(len(train_images)/n_batches)
    print(batch_size)
    while True:
        image_batch = train_images[batch_size* i: batch_size*(i+1)]
        label_batch = train_labels[batch_size* i: batch_size*(i+1)]
        yield((image_batch, to_categorical(label_batch)))
        i += 1
        if i == n_batches:
            i = 0

n_batches = 20
model.fit(
  generate_data(train_images, train_labels, n_batches),
  epochs=15,
  steps_per_epoch=n_batches,
  validation_data = (test_images, to_categorical(test_labels)),
  verbose=2,
)

# model.fit(
#   train_images,
#   to_categorical(train_labels),
#   epochs=15,
#   validation_split=0.05,
#   # validation_data = (test_images, to_categorical(test_labels)),
#   verbose=2,
# )

# loss, accuracy, f1_score, precision, recall = model.evaluate(test_images, to_categorical(test_labels), verbose=1)

model.save_weights(weights_dir + 'cnn_' + str(CLASS_ID) + '.h5')

print("done")
