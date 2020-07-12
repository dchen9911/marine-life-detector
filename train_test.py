import numpy as np
import glob
import matplotlib.image as mpimg
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras import backend as K
from consts import IMG_SIZE, CLASS_NAMES, CLASS_ID

class_name = CLASS_NAMES[CLASS_ID]

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
        if n_other > 1000:
            continue
        else:
            train_labels.append(0)
            train_images.append(mpimg.imread(filename))
            n_other += 1

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
  Conv2D(64, kernel_size=3, activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,3)),
  # Conv2D(32, kernel_size=3, activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,3)),
  # Conv2D(num_filters, filter_size, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
  MaxPooling2D(pool_size=2),
  Conv2D(64, kernel_size=3, activation='relu'),
  MaxPooling2D(pool_size=2),
  Conv2D(64, kernel_size=3, activation='relu'),
  MaxPooling2D(pool_size=2),
  Flatten(),
  #Dense(16, activation='relu'),
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
  epochs=20,
  # validation_split=0.1,
  validation_data = (test_images, to_categorical(test_labels)),
  verbose=2,
)

# loss, accuracy, f1_score, precision, recall = model.evaluate(test_images, to_categorical(test_labels), verbose=1)

model.save_weights('./weights/cnn_' + str(CLASS_ID) + '.h5')

print("done")
