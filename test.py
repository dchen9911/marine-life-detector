import glob
import shutil
import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf
import os
from consts import *

from CNN_model import model
from skimage import color
from skimage import io

class_name = CLASS_NAMES[CLASS_ID]

test_path = base_path + 'images/cropped_resized/train/'
fp_path = base_path + 'images/cropped_resized/false_neg/'
fn_path = base_path + 'images/cropped_resized/false_pos/'

# test_path = base_path + 'rcnn_test/'
# fp_path = base_path + 'false_neg/'
# fn_path = base_path + 'false_pos/'

if not os.path.exists(fp_path):
    os.mkdir(fp_path)
if not os.path.exists(fn_path):
    os.mkdir(fn_path)

weights_dir = base_path + 'weights/'

test_im = []
test_im_2 = []
pos_filepaths = []
neg_filepaths = []
n_max = 12000
n = 0
for filepath in glob.glob(test_path + '*.*'):
    n += 1
    if n > n_max:
        continue
    if USE_GREY:
        img = io.imread(filepath, as_gray=USE_GREY)
    else:
        img = mpimg.imread(filepath)
    if class_name not in filepath:
        test_im.append(img)
        neg_filepaths.append(filepath)
    else:
        test_im_2.append(img)
        pos_filepaths.append(filepath)
    


test_images = np.array(test_im)
test_images_2 = np.array(test_im_2)
test_images -= 0.5
test_images_2 -= 0.5
if USE_GREY:
    test_images = np.expand_dims(test_images, axis=3)
    test_images_2 = np.expand_dims(test_images_2, axis=3)

print(test_images.shape)

num_filters = 8
filter_size = 3

model.load_weights(weights_dir + 'cnn_' + str(CLASS_ID) + '.h5')

# test all the negative iamges
if len(test_images > 0):
    predictions = model.predict(test_images)
    # print(predictions)
    predictions = np.argmax(predictions, axis=1)
    pred_list = predictions.tolist()
    print("Misclassified {} out of {} negatives".format(pred_list.count(1), 
                                                len(pred_list)))
    for i in range(0, len(pred_list)):
        if pred_list[i] == 1:
            new_path = neg_filepaths[i].split('/')
            new_path[-2] = 'false_pos'
            new_path = '/'.join(new_path)
            shutil.copy(neg_filepaths[i], new_path)

# test all the positive images
if len(test_images_2 > 0):
    predictions = model.predict(test_images_2)
    # print(predictions)
    predictions = np.argmax(predictions, axis=1)
    pred_list = predictions.tolist()

    print("Misclassified {} out of {} positives".format(pred_list.count(0), 
                                                len(pred_list)))
    for i in range(0, len(pred_list)):
        if pred_list[i] == 0:
            new_path = pos_filepaths[i].split('/')
            new_path[-2] = 'false_neg'
            new_path = '/'.join(new_path)
            shutil.copy(pos_filepaths[i], new_path)
