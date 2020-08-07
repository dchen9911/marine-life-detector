import tensorflow as tf
from keras import callbacks
from keras import optimizers
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from consts import *
import glob
import numpy as np
import os 
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from skimage import color
from skimage import io
from sklearn.model_selection import train_test_split

from show_images import imageSet
from PIL import Image

original_weights = 'cnn_fine_earlystop.h5'

def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)
    bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
    bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

USE_PRELOADED = False

train_images = []
train_labels = []
class_name = CLASS_NAMES[CLASS_ID]
n_img_pos = 0
n_img_neg = 0 # for naming

if not USE_PRELOADED:
    # load the training data
    annot_file = base_path + 'crop_info_2.csv'
    image_dir = base_path + 'images/positive_raw/batch6/'        
    image_set = imageSet(image_dir, 200, shuffle=True)
    image_set.load_annotations_from_file(annot_file, file_format='csv')
    
    # put selective search images here  
    out_dir = base_path + 'images/fine_tune/fine_6/' 
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    

    # begin the selective search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # ax = plt.gca()
    for ind, key in enumerate(image_set.images_keys):
        print(ind)
        img = image_set.images_dict[key]
        print(img.img_name)
        if len(img.crop_areas) == 0:
            continue
        crop_areas = []
        for crop_area in img.crop_areas:
            x0,y0,x1,y1 = crop_area
            crop_areas.append({"x1":x0,"x2":x1,"y1":y0,"y2":y1})

        
        ss.setBaseImage(img.img)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()
        imout = img.img
        n_pos = 0
        n_neg = 0
        print(len(ssresults))
        for i, crop in enumerate(ssresults):
            
            if i > 2000 or (n_pos > 10 and n_neg > 10):
                break
            x,y,w,h = crop
            if w < MIN_SIZE or h < MIN_SIZE or w > MAX_SIZE or h > MAX_SIZE:
                continue

            pos_flag = 0 # 0 for negative sample, 1 for positive, -1 for dont accept
            for crop_area in crop_areas:
                iou = get_iou(crop_area, {"x1":x,"x2":x+w,"y1":y,"y2":y+h})
                
                if iou > 0.50:
                    pos_flag = 1
                    break
                elif iou > 0.2: # kinda a grey area so dont accept
                    pos_flag = -1

            if pos_flag == -1 :
                continue
            elif pos_flag == 0 and n_neg > 10:
                continue
            else:
                timage = imout[y:y+h,x:x+w]
                resized = cv2.resize(timage, (IMG_SIZE,IMG_SIZE), 
                                    interpolation = cv2.INTER_AREA)
                train_images.append(resized)
                train_labels.append(pos_flag)
                
                pil_img = Image.fromarray(resized)
                if pos_flag:
                    rect = Rectangle([x ,y], w, h, fill=False, color='orange', lw=1)
                    pil_img.save(out_dir + 'positive_' + str(n_img_pos) +'.png')
                    n_img_pos += 1
                    n_pos += 1
                else:
                    rect = Rectangle([x ,y], w, h, fill=False, color='yellow', lw=1)
                    pil_img.save(out_dir + 'negative_' + str(n_img_neg) +'.png')
                    n_img_neg += 1
                    n_neg += 1
            # ax.add_patch(rect)
    train_images = np.array(train_images)
    train_images = (train_images/255) - 0.5
    
else:
    fpaths = glob.glob(base_path + '/images/fine_tune/fine_1/' + '*.*')
    fpaths += glob.glob(base_path + '/images/fine_tune/fine_2/' + '*.*')
    fpaths += glob.glob(base_path + '/images/fine_tune/fine_3/' + '*.*')
    fpaths += glob.glob(base_path + '/images/fine_tune/fine_4/' + '*.*')
    fpaths += glob.glob(base_path + '/images/fine_tune/fine_5/' + '*.*')
    for filepath in fpaths:
        img = mpimg.imread(filepath)

        if class_name in filepath:
            train_labels.append(1)
            train_images.append(img)
            n_img_pos += 1
        else:
            train_labels.append(0)
            train_images.append(img)
            n_img_neg += 1

# train_images = np.array(train_images)
# train_images = train_images - 0.5
# train_labels = to_categorical(train_labels)

# X_train, X_test , y_train, y_test = train_test_split(train_images,train_labels,
#                                                       test_size=0.10)

# batch_size = 60

# trdata = ImageDataGenerator( 
#                     #horizontal_flip=True, vertical_flip=True, rotation_range=90
#                     )
# train_data = trdata.flow(x=X_train, y=y_train, batch_size=batch_size)
# tsdata = ImageDataGenerator( 
#                     #horizontal_flip=True, vertical_flip=True, rotation_range=90
#                             )
# test_data = tsdata.flow(x=X_test, y=y_test, batch_size=batch_size)

# from CNN_model import VGG_model

# print(train_images.shape)
# print(str(n_img_pos) + ' number of positives')
# print(str(n_img_neg) + ' number of negatives')

# # plt.show()
# VGG_model.load_weights(weights_dir + original_weights)

# for layer in (VGG_model.layers)[:15]:
#     layer.trainable = False

# VGG_model.compile(
#     optimizer=Adam(lr=0.001), 
#     loss='categorical_crossentropy', 
#     metrics=['accuracy']
# )


# check_dir = weights_dir + 'cnn_tuned_checkpoint.h5'
# checkpoint = ModelCheckpoint(check_dir, monitor='val_loss', 
#                              verbose=1, save_best_only=True, 
#                              save_weights_only=False, mode='auto')
# # We train it
# VGG_model.fit(train_data,
#           validation_data=test_data,
#           epochs=N_EPOCHS,
#           verbose=2,
#           callbacks=[checkpoint],
# )

# VGG_model.save_weights(weights_dir + 'cnn_tuned.h5')

