import matplotlib.image as mpimg
from matplotlib.widgets import Button

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
from PIL import Image
from consts import CLASS_NAMES
import pandas as pd
import json
import numpy as np
import os
import glob


# container class to store image data along with annotations
class imageContainer:
    def __init__(self, img, img_name):
        self.img = img # the actual image data
        self.img_name = img_name # filename of the image

        # the three lines below represent the annotation
        self.points = [] # list of [x,y] coords representing annotations
        self.polygons = [] # list of polygons
        self.class_ids = [] # list of class ids

        self.height, self.width, _ = img.shape
        self.rects = []
        self.bboxes = []
        self.crop_areas = [] # tuple of (x0, y0, x1, y1) coords

    def add_annot(self, coord, polygon, class_id=0, prescaled=False):
        if not prescaled:
            x_coord = self.width*coord[0] 
            y_coord = self.height*coord[1]
            self.points.append([x_coord, y_coord])

            polygon = np.array(polygon)
            polygon[:,1] *= self.height
            polygon[:,1] += y_coord
            polygon[:,0] *= self.width
            polygon[:,0] += x_coord

            # x1 = int(min(polygon[:,0]))
            # y1 = int(min(polygon[:,1]))
            # width = int(max(polygon[:,0]) - x1)
            # height = int(max(polygon[:,1]) - y1)
            self.class_ids.append(class_id)
            if class_id == 0:
                col = 'red'
            else:
                col = 'blue'
            polygon = Polygon(polygon, alpha=0.4, lw=0.5, fill=False, color=col)

            self.polygons.append(polygon)  

            bbox = polygon.get_extents()
            self.bboxes.append(bbox)

            x0, y0, width, height = bbox.bounds
            self.rects.append(Rectangle([x0 ,y0], width, height, fill=False, 
                                         color=col, lw=2)
                              )
        else:
            raise NotImplementedError("add_annot")

# another container class that contains all data for images in given folder
# and loads all images into containers
class imageSet:
    # img_dir: string of directory where all images are stored
    # filename: string of filename where all the image names are stored
    def __init__(self, img_dir, img_info_file_name=None, size_limit=100):
        self.images_dict = {}
        self.size_limit = size_limit
        self.img_dir = img_dir

        if img_info_file_name is not None:
            self.load_images_from_file(img_info_file_name)
        else:
            self.load_images_from_dir()

        self.i = 0 # index of image in images_list that we are currently 
                   # working with 
        self.images_list = [(k, v) for k, v in self.images_dict.items()]
        
        self.prev_sel = None # tuple of values that was previously selected 
        self.crop_n = 355
        self.crop_info_file = None # file object where crop box data is appended

    def load_images_from_dir(self):
        fpaths = []
        fpaths += glob.glob(self.img_dir + '*.jpg')
        fpaths += glob.glob(self.img_dir + '*.png')
        fpaths.sort()
        for fpath in fpaths:
            img_name = fpath.split('/')[-1]
            new_img = imageContainer(mpimg.imread(fpath), img_name)
            self.images_dict[img_name] = new_img
        
    def load_images_from_file(self, filename):
        images_dict = self.images_dict
        df = pd.read_csv(filename)
        for index, row in df.iterrows():
            if len(images_dict) > self.size_limit:
                break
            img_name = row['media_key'].strip()
            if img_name not in images_dict.keys():
                try:
                    new_img = imageContainer(mpimg.imread(self.img_dir + img_name),
                                             img_name)
                    images_dict[img_name] = new_img
                except FileNotFoundError:
                    print("File " + img_name + ' not found in' + self.img_dir)
                    continue
        
            coord = [float(row['x']), float(row['y'])]
            polygon = json.loads(row['point_data'])['polygon']
            try:
                class_id = CLASS_NAMES.index(row['class_name'])
            except ValueError:
                class_id = -1
            images_dict[img_name].add_annot(coord, polygon, class_id)

    def next_image(self, event):
        self.i += 1
        if self.i >= len(self.images_list):
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

    def set_crop_save_file(self, file):
        self.crop_info_file = file

    def set_crop_img_dest(self, dest):
        self.crop_img_dir = dest

    def save_crop_boxes(self, event):
        img = self.images_list[self.i][1]
        if self.crop_info_file is not None:
            areas_dict = {'crop_areas': img.crop_areas}
            self.crop_info_file.write(img.img_name + ',' + '\"' + 
                                      json.dumps(areas_dict) + '\"' + '\n')
        for crop_area in img.crop_areas:
            print(crop_area)

            img_arr = img.img
            # TODO: Uncomment if its PNG
            if img.img_name.split('.')[-1] == 'png':
                pil_img = Image.fromarray(np.uint8(img_arr*255))
            else:
                pil_img = Image.fromarray(img_arr)
            cropped_img = pil_img.crop(crop_area)
            file_path = self.crop_img_dir + 'crop_' + str(self.crop_n) + '.jpg'
            print("saving image to " + file_path)
            cropped_img.save(file_path)
            self.crop_n += 1
        
        self.next_image(None)

    def set_plot_ax(self, ax):
        self.ax = ax

    def set_plot_fig(self, fig):
        self.fig = fig

    def add_crop_box(self, p1, p2):
        x0 = min(p1[0], p2[0])
        y0 = min(p1[1], p2[1])
        x1 = max(p1[0], p2[0])
        y1 = max(p1[1], p2[1])
        img = self.images_list[self.i][1]
        
        img.crop_areas.append((x0, y0, x1, y1))
        return x0, y0, x1-x0, y1-y0

    # displays the ith image
    def display_image(self):
        self.ax.clear()
        self.prev_sel = None
        img = self.images_list[self.i][1]
        self.ax.imshow(img.img)

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

if __name__ == "__main__":
    image_dir = './cuttlefish_data/images/positive_5/'
    crop_img_dir = './cuttlefish_data/images/positive_crop/'
    if not os.path.exists(crop_img_dir):
        os.mkdir(crop_img_dir)

    img_info_file = './sesoko_data/sesoko_crest.csv'
    image_set = imageSet(image_dir, 
                        #filename=img_info_file, 
                         size_limit=100)

    image_set.set_crop_img_dest(crop_img_dir)
    store_info_file = open("./cuttlefish_data/crop_info.csv", "a+")
    
    image_set.set_crop_save_file(store_info_file)

    ax = plt.gca()
    fig = plt.gcf()
    image_set.set_plot_ax(ax)
    image_set.set_plot_fig(fig)

    axprev = plt.axes([0.25, 0.9, 0.1, 0.05])
    axsave = plt.axes([0.45, 0.9, 0.1, 0.05])
    axnext = plt.axes([0.65, 0.9, 0.1, 0.05])
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(image_set.prev_image)
    bsave = Button(axsave, 'Save')
    bsave.on_clicked(image_set.save_crop_boxes)
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(image_set.next_image)
    


    image_set.display_image()
    
    cid = fig.canvas.mpl_connect('button_press_event', image_set.onclick)
    plt.show()
        
    #store_info_file.close()
    
    # 
        # self.ax.add_collection(pc)
        # self.patch_collections.append(pc)

    
