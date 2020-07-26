import os
import glob
import random
import shutil
from PIL import Image, ImageOps
from consts import IMG_SIZE, CLASS_NAMES, base_path

SPLIT = 9 # 1/SPLIT fraction gets set as test set

def split_images(src, dest, split=9, limit=11000, copy=False, to_grey=False):
    if not os.path.exists(dest):
        os.mkdir(dest)
    test_dest = dest + 'test/'
    train_dest = dest + 'train/'

    if not os.path.exists(test_dest):
        os.mkdir(test_dest)
    if not os.path.exists(train_dest):
        os.mkdir(train_dest)
    fpaths = glob.glob(src + '*.*')
    random.shuffle(fpaths)
    fpaths = fpaths[0:limit]
    for i, filepath in enumerate(fpaths):
        filename = filepath.split('/')[-1]
        if i % split == 0:
            final_dest = test_dest + filename
        else:
            final_dest = train_dest + filename
        
        if to_grey:
            im=Image.open(filepath).convert('LA')
            im.save(final_dest)
        elif copy:
            shutil.copyfile(filepath, final_dest)
        else:
            shutil.move(filepath, final_dest)

def unsplit_images(src, dest, split=9, limit=10000, to_grey=False):
    test_dest = dest + 'test/'
    train_dest = dest + 'train/'

    fpaths = glob.glob(test_dest + '*.*')
    fpaths += glob.glob(train_dest + '*.*')
    for i, filepath in enumerate(fpaths):
        if 'negative' in filepath:
            filename = filepath.split('/')[-1]
            if to_grey:
                os.remove(filepath)
            else:
                shutil.move(filepath, src + filename)
    
# split every x images
def resize_images(src, dest, split=None, to_grey=False):
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
        if to_grey:
            new_im = new_im.convert('LA')
        name = filename.split('/')[-1]     
        name = name.split('.')[0]

        if split:
            if i % split == 0:
                final_dest = test_dest
            else:
                final_dest = train_dest
            i += 1
            
        for x in range(4):
            save_im = new_im.rotate(90*x)
            save_im.save(final_dest + name + '_' + str(x) + '.png')
        im_flip = ImageOps.flip(new_im)
        im_flip.save(final_dest + name + '_4' + '.png')
        im_mirror = ImageOps.mirror(new_im)
        im_mirror.save(final_dest + name + '_5' + '.png')
        

def rename_files():
    src_dir = './cuttlefish_data/images/positive_crop/'
    for filepath in glob.glob(src_dir + '*.*'):
        file_arr = filepath.split('/')
        file_arr[-1] = 'positive_' + file_arr[-1]
        new_filepath = '/'.join(file_arr)
        os.rename(filepath, new_filepath)

if __name__ == "__main__":
    src = base_path + 'images/negative_crop/'
    dest =  base_path + 'images/cropped_resized/'

    split_images(src, dest, split=SPLIT, limit=12500)

    src =  base_path + 'images/positive_crop/'

    resize_images(src, dest, split=SPLIT)




