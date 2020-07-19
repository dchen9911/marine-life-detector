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
    fnames = glob.glob(src + '*.*')
    random.shuffle(fnames)

    i = 0
    final_dest = dest
    
    for filename in fnames:
        im=Image.open(filename)
        new_im = im.resize((IMG_SIZE, IMG_SIZE))
        name = filename.split('/')[-1]     
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

def rename_files():
    src_dir = './cuttlefish_data/images/positive_crop/'
    for filepath in glob.glob(src_dir + '*.*'):
        file_arr = filepath.split('/')
        file_arr[-1] = 'positive_' + file_arr[-1]
        new_filepath = '/'.join(file_arr)
        os.rename(filepath, new_filepath)

if __name__ == "__main__":
    src = './cuttlefish_data/images/negative_crop/'
    dest = './cuttlefish_data/images/cropped_resized/'

    resize_images(src, dest, split=SPLIT)

    src = './cuttlefish_data/images/positive_crop/'
    dest = './cuttlefish_data/images/cropped_resized/'

    resize_images(src, dest, split=SPLIT)




