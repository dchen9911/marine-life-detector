import tensorflow as tf
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

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
train_path = base_path + 'images/crops/resized/'

train_images = []
train_labels = []

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
        
if USE_GREY:
    train_images = np.expand_dims(train_images, axis=3)

print('Positive examples: ' + str(len(train_images) - n_other) + '/' + str(len(train_images)))
train_labels = np.array(train_labels)
train_images = np.array(train_images)
train_images -= 0.5

print(train_images.shape)

train_labels = to_categorical(train_labels)

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

X_train, X_test , y_train, y_test = train_test_split(train_images,train_labels,
                                                      test_size=0.10)

trdata = ImageDataGenerator()
train_data = trdata.flow(x=X_train, y=y_train, batch_size=batch_size)
tsdata = ImageDataGenerator()
test_data = tsdata.flow(x=X_test, y=y_test, batch_size=batch_size)

check_dir = weights_dir + 'cnn_checkpoint.h5'


checkpoint = ModelCheckpoint(check_dir, monitor='val_loss', 
                             verbose=1, save_best_only=True, 
                             save_weights_only=False, mode='auto', period=1)

# We train it
VGG_model.fit(train_data,
          validation_data=test_data,
          epochs=N_EPOCHS,
          verbose=2,
          callbacks=[checkpoint],
)

VGG_model.save_weights(weights_dir + 'cnn_' + str(CLASS_ID) + '.h5')

