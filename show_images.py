import pandas as pd
import json
import numpy as np
import os
import glob
import argparse
import random
from PIL import Image
from consts import *

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from matplotlib.widgets import Button
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection


# container class to store image data along with annotations
class imageContainer:
    def __init__(self, img, img_name):
        self.img = img # the actual image data
        self.img_name = img_name # filename of the image

        # the four vars below represent the annotation
        self.points = [] # list of [x,y] coords representing point annotations (optional)
        self.polygons = [] # list of polygons (optional)
        self.class_ids = [] # list of class ids (required)
        self.rects = [] # list of matplotlib.patches.Rectangle objects (required)

        self.height, self.width, _ = img.shape   

        self.crop_areas = [] # tuple of (x0, y0, x1, y1) coords

    # norm: whether or not to normalise the coordinates to the height and width
    def add_annot(self, coord, class_id=0, polygon=None, norm=True):
        self.class_ids.append(class_id)
        if class_id == 0:
            col = 'red'
        else:
            col = 'blue'

        if norm and polygon:
            x_coord = self.width*coord[0] 
            y_coord = self.height*coord[1]
            self.points.append([x_coord, y_coord])

            polygon = np.array(polygon)
            polygon[:,1] *= self.height
            polygon[:,1] += y_coord
            polygon[:,0] *= self.width
            polygon[:,0] += x_coord

            polygon = Polygon(polygon, lw=1.5, fill=False, color='green')
            print("adding polygon")
            self.polygons.append(polygon)  

            bbox = polygon.get_extents()
            x0, y0, width, height = bbox.bounds
        else:
            assert len(coord) == 4, 'this type of annotation unimplemented'
            x0, y0, x1, y1 = coord
            width = x1 - x0
            height = y1 - y0

        crop_area = (x0, y0, x0 + width, y0 + height)
        self.crop_areas.append(crop_area)
        self.rects.append(Rectangle([x0 ,y0], width, height, fill=False, 
                                color=col, lw=2)
                         )

# another container class that contains all data for images in given folder
# and loads all images into containers
class imageSet:
    # img_dir: string of directory where all images are stored
    # filename: string of filename where all the image names are stored
    def __init__(self, img_dir, size_limit=1000, crop_n=0, load=True, shuffle=False):
        self.images_dict = {}
        self.size_limit = size_limit
        self.img_dir = img_dir
        if load:
            self.load_images_from_dir(img_dir, shuffle)
            self.images_keys = list(self.images_dict.keys())           

        self.i = 0 # index of image in images_list that we are currently 
                   # working with 
        
        # variables for the plotting stuff
        self.ax = None
        self.prev_sel = None # tuple of values that was previously selected 
        self.crop_n = crop_n
        
        self.crop_info_filepath = None # filename crop box data is appended
        self.crop_img_dest = None # crop img_dir

    def load_images_from_dir(self, img_dir, shuffle=False):
        fpaths = glob.glob(img_dir + '*.*')
        if shuffle:
            random.shuffle(fpaths)
        else:
            fpaths.sort()
        for i, fpath in enumerate(fpaths):
            if i >= self.size_limit:
                break
            img_name = fpath.split('/')[-1]
            new_img = imageContainer(mpimg.imread(fpath), img_name)
            self.images_dict[img_name] = new_img
        
    def load_annotations_from_file(self, filepath, file_format='df'):
        images_dict = self.images_dict
        if file_format == 'df':
            df = pd.read_csv(filepath)
            for index, row in df.iterrows():
                img_name = row['media_key'].strip()
                if img_name not in images_dict.keys():
                    print("File " + img_name + ' not found in' + self.img_dir)
                    continue
                coord = [float(row['x']), float(row['y'])]
                polygon = json.loads(row['point_data'])['polygon']
                try:
                    class_id = CLASS_NAMES.index(row['class_name'])
                except ValueError:
                    class_id = -1
                images_dict[img_name].add_annot(coord, class_id, polygon)
        elif file_format == 'row':
            fp = open(filepath, 'r')
            for line in fp.readlines():
                img_name, x0, y0, x1, y1, _ = line.split(',')
                img_name = img_name.split('/')[-1]
                if img_name not in images_dict.keys():
                    # print("File " + img_name + ' not found in' + self.img_dir)
                    continue
                crop_area = list(map(int, [x0, y0, x1, y1])) 
                images_dict[img_name].add_annot(crop_area, 0)
                images_dict[img_name].crop_areas.append(crop_area)
            fp.close()
        elif file_format == 'csv':
            df = pd.read_csv(filepath)
            for index, row in df.iterrows():
                img_name = row['name']
                if img_name not in images_dict.keys():
                    # print("File " + img_name + ' not found in' + self.img_dir)
                    continue
                crop_areas = json.loads(row['area'])['crop_areas']
                for ca in crop_areas:
                    images_dict[img_name].add_annot(ca, 0)
                    images_dict[img_name].crop_areas.append(ca)
    # sets where the crop outputs go and whether or not they should be written to 
    # a file
    def set_crop_output(self, dest, info_file_path=None):
        if not os.path.exists(dest):
            os.mkdir(dest)
        self.crop_img_dest = dest 
        if info_file_path is not None:
            self.crop_info_filepath = info_file_path

    # take some random sized crops of given image set
    def crop_random(self, n=3, min_size=20, max_size=600, 
                    filter_dark=False, resize=False):
        assert self.crop_img_dest is not None, "Set crop output first"

        dest_dir = self.crop_img_dest
        if self.crop_info_filepath is not None:
            info_file = open(crop_info_file, "w+")
        
        fpaths = glob.glob(self.img_dir + '*.*')
        fpaths.sort()
        for ind, fpath in enumerate(fpaths):
            if ind >= self.size_limit:
                break
            name = fpath.split('/')[-1]
            img = imageContainer(mpimg.imread(fpath), name)
            im_height, im_width, _ = img.img.shape
            if filter_dark:
                avg_col = np.mean(img.img)
                if avg_col < 70:
                    continue

            for i in range(n):
                crop_height = random.randint(min_size, max_size)
                crop_width = random.randint(min_size, max_size)

                start_x = random.randint(0, im_width - crop_width)
                start_y = random.randint(0, im_height - crop_height)
                end_x = start_x + crop_width
                end_y = start_y + crop_height

                pil_img = Image.fromarray(img.img)
                crop_area = (start_x, start_y, end_x, end_y)
                cropped_img = pil_img.crop(crop_area)
                if resize:
                    cropped_img = cropped_img.resize((IMG_SIZE, IMG_SIZE))
                fname = 'negative_' + str(self.crop_n) +'.png'
                cropped_img.save(dest_dir + fname)
                out_str = '{},{},{},{},{},{}\n'.format(fname, img.img_name, 
                                            start_x, start_y, end_x, end_y)
                info_file.write(out_str)
                self.crop_n += 1

    # crop annots: crops out the bboxes
    def crop_annots(self, dest_dir):
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)

        name_dict = {}
        for name, img in self.images_dict.items():
            for class_id, crop_area in zip(img.class_ids, img.crop_areas):
                img_arr = img.img
                if img.img_name.endswith('.png'):
                    pil_img = Image.fromarray(np.uint8(img_arr*255))
                else:
                    pil_img = Image.fromarray(img_arr)
                cropped_img = pil_img.crop(crop_area)

                if class_id == -1:
                    name = 'other'
                elif class_id == 2: # for foliose
                    name = 'foliose'
                else:
                    name = CLASS_NAMES[class_id]

                if name not in name_dict.keys():
                    name_dict[name] = 1
                else:
                    name_dict[name] += 1
                name_str = dest_dir + name + '_' + str(name_dict[name]) + '.png'
                cropped_img.save(name_str)


    def setup_plot(self):
        ax = plt.gca()
        fig = plt.gcf()
        self.ax = ax

        axprev = plt.axes([0.8, 0.8, 0.1, 0.05])
        axclear = plt.axes([0.8, 0.6, 0.1, 0.05])
        axsave = plt.axes([0.8, 0.4, 0.1, 0.05])
        axnext = plt.axes([0.8, 0.2, 0.1, 0.05])

        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(self.prev_image)
        bclear = Button(axclear, 'Clear')
        bclear.on_clicked(self.clear_annots)
        bsave = Button(axsave, 'Save')
        bsave.on_clicked(self.save_crop_boxes)
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(self.next_image)

        image_set.display_image()
        
        cid = fig.canvas.mpl_connect('button_press_event', image_set.onclick)
        plt.show()

    def add_crop_box(self, p1, p2):
        x0 = min(p1[0], p2[0])
        y0 = min(p1[1], p2[1])
        x1 = max(p1[0], p2[0])
        y1 = max(p1[1], p2[1])
        img = self.images_dict[self.images_keys[self.i]]
        
        img.crop_areas.append([x0, y0, x1, y1])
        return x0, y0, x1-x0, y1-y0

    def next_image(self, event):
        self.i += 1
        if self.i >= len(self.images_keys):
            print('Finished all images, exiting...')
            exit(0)
        self.display_image()
    
    def prev_image(self, event):
        self.i -= 1
        if self.i < 0:
            print('Cant go any further back, error')
            self.i = 0
            return
        self.display_image()

    def clear_annots(self, event):
        img = self.images_dict[self.images_keys[self.i]]
        img.rects = []
        img.crop_areas = []
        img.class_ids = []
        self.display_image()

    # displays the ith image
    def display_image(self):
        assert self.ax is not None, 'Plot needs to first be set up'
        self.ax.clear()
        self.prev_sel = None
        print(self.i)
        img = self.images_dict[self.images_keys[self.i]]
        self.ax.imshow(img.img)
        if len(img.points) == 2:
            x_coords = [point[0] for point in img.points]
            y_coords = [point[1] for point in img.points]
            self.ax.plot(x_coords, y_coords, 'bo')

        for polygon in img.polygons:
            self.ax.add_patch(polygon)

        for rect in img.rects:
            self.ax.add_patch(rect)
        plt.draw()
    
    def onclick(self, event):
        if event.inaxes != self.ax:
            return
        curr_sel = [event.xdata, event.ydata]
        if not self.prev_sel:
            self.prev_sel = curr_sel
        else:
            print("adding rectangle")
            x0, y0, w, h = self.add_crop_box(self.prev_sel, curr_sel)
            rect_patch = Rectangle([x0, y0], w, h, fill=False, 
                                   color='green', lw=1)
            self.ax.add_patch(rect_patch)
            self.prev_sel = None
            plt.draw()

    def save_crop_boxes(self, event):
        assert self.crop_img_dest is not None, "Set crop output first"

        img = self.images_dict[self.images_keys[self.i]]
        if self.crop_info_filepath is not None:
            info_file = open(self.crop_info_filepath, "a+")
            areas_dict = {'crop_areas': img.crop_areas}
            info_file.write(img.img_name + ',' + '\"' + 
                                      json.dumps(areas_dict) + '\"' + '\n')
        for crop_area in img.crop_areas:
            img_arr = img.img
            if img.img_name.endswith('.png'):
                pil_img = Image.fromarray(np.uint8(img_arr*255))
            else:
                pil_img = Image.fromarray(img_arr)
            cropped_img = pil_img.crop(crop_area)
            file_path = self.crop_img_dest + 'positive_' + str(self.crop_n) + '.png'
            print("saving image to " + file_path)
            cropped_img.save(file_path)
            self.crop_n += 1
        
        self.next_image(None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify what to do')
    parser.add_argument('-a', '--annotate', required=False, 
                        help='Annotate some images', action='store_true')
    parser.add_argument('-cr', '--crop_random', required=False,
                        help="Do some random crops", action='store_true')
    parser.add_argument('-v', '--view', required=False,
                        help="View annotated images", action='store_true')
    parser.add_argument('-ca', '--crop_annots', required=False,
                        help="Crop annotations images", action='store_true')
    parser.add_argument('-ma', '--modify_annots', required=False,
                        help="Modify image annotations", action='store_true')
    args = vars(parser.parse_args())

    image_dir = base_path + 'images/positive_7/'

    annot_file = base_path + 'annots.csv' # file where annotations are stored
    crop_dest = base_path + 'images/positive_crop/' # destination of cropped images

    if args['annotate']:
        image_dir = base_path + 'images/positive_9/'        
        image_set = imageSet(image_dir, 
                         size_limit=110,
                         crop_n=608) # FIXME: Make sure to choose crop_n correctly
    
        crop_info_file = base_path + 'crop_info.csv'
        crop_dest = base_path + 'images/positive_crop/' 
        image_set.set_crop_output(crop_dest, crop_info_file)
        image_set.setup_plot()

    elif args['modify_annots']:
        image_dir = base_path + 'images/positive_raw/'        
        image_set = imageSet(image_dir, 
                         size_limit=1000,
                         crop_n=0)
        annot_file = base_path + 'crop_info.csv' # file where annotations are stored
    
        image_set.load_annotations_from_file(annot_file, file_format='csv')
        crop_info_file = base_path + 'crop_info_2.csv'
        crop_dest = base_path + 'images/positive_crop_2/' 
        image_set.set_crop_output(crop_dest, crop_info_file)
        image_set.setup_plot()
    elif args['crop_random']:
        image_dir = base_path + 'images/negative_raw/'
        image_set = imageSet(image_dir, 
                            load=False,
                            size_limit=15000,
                            crop_n=0)

        crop_dest = base_path + 'images/negative_crop/'
        crop_info_file = base_path + 'neg_crop_info.csv'
        image_set.set_crop_output(crop_dest, crop_info_file)
        image_set.crop_random(7, MIN_SIZE, 600, resize=True)

    elif args['crop_annots']:
        image_dir = base_path + 'images/positive_raw/'
        image_set = imageSet(image_dir, size_limit=1000)
        image_set.load_annotations_from_file(annot_file, file_format='row')
        image_set.crop_annots(crop_dest)

    elif args['view']:
        image_dir = base_path + 'images/raw/'
        image_set = imageSet(image_dir, size_limit=10)
        image_set.load_annotations_from_file(annot_file, file_format='df')
        image_set.setup_plot()
    
