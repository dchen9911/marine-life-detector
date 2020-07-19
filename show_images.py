import pandas as pd
import json
import numpy as np
import os
import glob
import argparse
import random
from PIL import Image
from consts import CLASS_NAMES, base_path

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

            polygon = Polygon(polygon, alpha=0.4, lw=0.5, fill=False, color=col)

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
    def __init__(self, img_dir, size_limit=1000, crop_n=0):
        self.images_dict = {}
        self.size_limit = size_limit
        self.img_dir = img_dir
        self.load_images_from_dir(img_dir)           

        self.i = 0 # index of image in images_list that we are currently 
                   # working with 
        self.images_keys = list(self.images_dict.keys())
        
        # variables for the plotting stuff
        self.ax = None
        self.prev_sel = None # tuple of values that was previously selected 
        self.crop_n = crop_n
        self.crop_info_file = None # file object where crop box data is appended
        self.crop_img_dest = None # crop img_dir

    def load_images_from_dir(self, img_dir):
        fpaths = glob.glob(img_dir + '*.*')
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
                    print("File " + img_name + ' not found in' + self.img_dir)
                    continue
                coords = list(map(int, [x0, y0, x1, y1])) 
                images_dict[img_name].add_annot(coords, 0)
                images_dict[img_name].crop_areas.append(coords)
            fp.close()

    # take some random sized crops of given image set
    def crop_random(self, dest_dir, min_size=200, max_size=600):
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        n = 0
        for name, img in self.images_dict.items():
            im_height, im_width, _ = img.img.shape
            for i in range(2):
                crop_height = random.randint(min_size, max_size)
                crop_width = random.randint(min_size, max_size)
                start_x = random.randint(0, im_width - crop_width)
                start_y = random.randint(0, im_width - crop_height)
                pil_img = Image.fromarray(img.img)
                crop_area = (start_x, start_y, start_x + crop_width, 
                            start_y + crop_height)
                cropped_img = pil_img.crop(crop_area)
                cropped_img.save(dest_dir + 'negative_' + str(n) +'.png')
                n += 1

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

        axprev = plt.axes([0.25, 0.9, 0.1, 0.05])
        axsave = plt.axes([0.45, 0.9, 0.1, 0.05])
        axnext = plt.axes([0.65, 0.9, 0.1, 0.05])
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(self.prev_image)
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
        
        img.crop_areas.append((x0, y0, x1, y1))
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

    # displays the ith image
    def display_image(self):
        assert self.ax is not None, 'Plot needs to first be set up'
        self.ax.clear()
        self.prev_sel = None
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

    # sets where the crop outputs go and whether or not they should be written to 
    # a file
    def set_crop_output(self, dest, info_file_path=None):
        if not os.path.exists(crop_img_dest):
            os.mkdir(crop_img_dest)
        self.crop_img_dest = dest 
        self.crop_info_file = open(info_file_path, "a+")

    def save_crop_boxes(self, event):
        assert self.crop_img_dest is not None, "Set crop output first"

        img = self.images_dict[self.images_keys[self.i]]
        if self.crop_info_file is not None:
            areas_dict = {'crop_areas': img.crop_areas}
            self.crop_info_file.write(img.img_name + ',' + '\"' + 
                                      json.dumps(areas_dict) + '\"' + '\n')
        for crop_area in img.crop_areas:
            img_arr = img.img
            if img.img_name.endswith('.png'):
                pil_img = Image.fromarray(np.uint8(img_arr*255))
            else:
                pil_img = Image.fromarray(img_arr)
            cropped_img = pil_img.crop(crop_area)
            file_path = self.crop_img_dest + 'positive_crop_' + str(self.crop_n) + '.png'
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
    args = vars(parser.parse_args())

    image_dir = base_path + 'images/positive_6/'

    annot_file = base_path + 'annots.csv' # file where annotations are stored
    crop_dest = base_path + 'images/positive_crop/' # destination of cropped images

    if args['annotate']:
        image_set = imageSet(image_dir, 
                         # annot_file=annot_file, 
                         size_limit=110,
                         crop_n=450)
    
        crop_info_file = base_path + 'crop_info.csv'
        image_set.set_crop_output(crop_dest, crop_info_file)
        image_set.setup_plot()
    elif args['crop_random']:
        image_dir = base_path + 'images/negative_raw/'
        image_set = imageSet(image_dir, 
                            # annot_file=annot_file, 
                            size_limit=1000,
                            crop_n=450)
        image_set.crop_random(crop_dest)

    elif args['crop_annots']:
        image_dir = base_path + 'images/positive_raw/'
        image_set = imageSet(image_dir, size_limit=1000)
        image_set.load_annotations_from_file(annot_file, file_format='row')
        image_set.crop_annots(crop_dest)

    elif args['view']:
        image_dir = base_path + 'images/positive_raw/'
        image_set = imageSet(image_dir, size_limit=1000)
        image_set.load_annotations_from_file(annot_file, file_format='row')
        image_set.setup_plot()
    
