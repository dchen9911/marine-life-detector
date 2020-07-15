import numpy as np
import glob
import matplotlib.image as mpimg
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras import backend as K
from consts import IMG_SIZE, CLASS_NAMES, CLASS_ID

class_name = CLASS_NAMES[CLASS_ID]

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# load the training data
train_path = './images/cropped_resized/train/'
train_images = []
train_labels = []

n_other = 0
for filename in glob.glob(train_path + '*.png'):
    if class_name in filename:
        train_labels.append(1)
        train_images.append(mpimg.imread(filename))
    else:
        if CLASS_ID == 0:
            n_nmax = 10000
        else:
            n_max = 1000
        if n_other > n_nmax:
            continue
        else:
            train_labels.append(0)
            train_images.append(mpimg.imread(filename))
            n_other += 1
print(len(train_images) - n_other)
train_labels = np.array(train_labels)
train_images = np.array(train_images)
train_images -= 0.5

print(train_images.shape)

# load test data
test_path = './images/cropped_resized/test/'
test_images = []
test_labels = []

for filename in glob.glob(test_path + '*.png'):
    test_images.append(mpimg.imread(filename))
    if class_name in filename:
        test_labels.append(1)
    else:
        test_labels.append(0)

test_labels = np.array(test_labels)
test_images = np.array(test_images)
test_images -= 0.5

print(test_images.shape)

num_filters = 8
filter_size = 3
pool_size = 3

model = Sequential([
  Conv2D(8, kernel_size=3, activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,3)),
  MaxPooling2D(pool_size=2),
  Conv2D(16, kernel_size=3, activation='relu'),
  MaxPooling2D(pool_size=2),
  Conv2D(32, kernel_size=3, activation='relu'),
  MaxPooling2D(pool_size=2),
  Dropout(0.5),
  Flatten(),
  Dense(128, activation='relu'),
  # Dropout(0.5),
  Dense(2, activation='softmax'),
])

model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=15,
  # validation_split=0.1,
  validation_data = (test_images, to_categorical(test_labels)),
  verbose=2,
)

# loss, accuracy, f1_score, precision, recall = model.evaluate(test_images, to_categorical(test_labels), verbose=1)

model.save_weights('./weights/cnn_' + str(CLASS_ID) + '.h5')

print("done")
