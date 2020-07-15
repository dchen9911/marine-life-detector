import os
import glob
import random
from PIL import Image
from consts import IMG_SIZE, CLASS_NAMES

SPLIT = 7 # 1/SPLIT fraction gets set as test set

# split every x images
def resize_images(src, dest, split=None):
    if not os.path.exists(dest):
        os.mkdir(dest)
    if split:
        test_dest = dest + 'test/'
        train_dest = dest + 'train/'
        if not os.path.exists(test_dest):
            os.mkdir(test_dest)
        if not os.path.exists(train_dest):
            os.mkdir(train_dest)

    image_list = []
    names = []

    fnames = glob.glob(src + '*.png')
    random.shuffle(fnames)

    for filename in fnames:
        im=Image.open(filename)
        image_list.append(im)
        names.append(filename.split('/')[-1])

    i = 0
    final_dest = dest
    for im, name in zip(image_list, names):
        new_im = im.resize((IMG_SIZE, IMG_SIZE))
        name = name.split('.')[0]
        if split:
            if i % 6 == 0:
                final_dest = test_dest
            else:
                final_dest = train_dest
            i += 1
        for x in range(4):
            save_im = new_im.rotate(90*x)
            save_im.save(final_dest + name + '_' + str(x) + '.png')

        
if __name__ == "__main__":
    src = './sesoko_data/images/cropped/'
    dest = './sesoko_data/images/cropped_resized/'

    resize_images(src, dest, split=SPLIT)




