from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import glob
import numpy as np
import matplotlib.image as mpimg
from consts import IMG_SIZE, CLASS_NAMES, CLASS_ID

class_name = CLASS_NAMES[CLASS_ID]

test_path = './images/cropped_resized/test/'
test_images = []
test_images_2 = []

for filename in glob.glob(test_path + '*.png'):
    if class_name not in filename:
        test_images.append(mpimg.imread(filename))
    else:
        test_images_2.append(mpimg.imread(filename))


test_images = np.array(test_images)
test_images_2 = np.array(test_images_2)
test_images -= 0.5
test_images_2 -= 0.5

print(test_images.shape)

num_filters = 8
filter_size = 3

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
  # Dense(16, activation='relu'),
  Dense(2, activation='softmax'),
])


model.load_weights('./weights/cnn_' + str(CLASS_ID) + '.h5')

predictions = model.predict(test_images)
predictions = np.argmax(predictions, axis=1)
print("Misclassified {} out of {} negatives".format(predictions.tolist().count(1), 
                                                    len(predictions)))

predictions = model.predict(test_images_2)
predictions = np.argmax(predictions, axis=1)
print("Misclassified {} out of {} positives".format(predictions.tolist().count(0), 
                                                    len(predictions)))
