from PIL import Image
import os
import glob
from consts import IMG_SIZE, CLASS_NAMES

SPLIT = 6 # 1/SPLIT fraction gets set as test set

src = './images/cropped/'
dest = './images/cropped_resized/'
test_dest = './images/cropped_resized/test/'
train_dest = './images/cropped_resized/train/'

if not os.path.exists(dest):
    os.mkdir(dest)
if not os.path.exists(test_dest):
    os.mkdir(test_dest)
if not os.path.exists(train_dest):
    os.mkdir(train_dest)

image_list = []
names = []

for filename in glob.glob(src + '*.png'):
    im=Image.open(filename)
    image_list.append(im)
    names.append(filename.split('/')[-1])

i = 0
for im, name in zip(image_list, names):
    new_im = im.resize((IMG_SIZE, IMG_SIZE))
    name = name.split('.')[0]

    if i % 6 == 0:
        final_dest = test_dest
    else:
        final_dest = train_dest
    for x in range(4):
        save_im = new_im.rotate(90*x)
        save_im.save(final_dest + name + '_' + str(x) + '.png')

    i += 1

