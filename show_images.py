import matplotlib.image as mpimg

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
from PIL import Image
from consts import CLASS_NAMES
import pandas as pd
import json
import numpy as np
import os

# container class to store image data along with annotations
class imageContainer:
    def __init__(self, img):
        self.img = img # the actual image data
        self.points = [] # list of [x,y] coords representing annotations
        self.polygons = [] # list of polygons
        self.class_ids = [] # list of class ids
        self.height, self.width, _ = img.shape
        self.rects = []
        self.bboxes = []

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
    def __init__(self, filepath):
        self.df = pd.read_csv('sesoko_crest.csv')
        self.images_dict = {}
        self.load_images()

    def load_images(self):
        images_dict = self.image_dict
        for index, row in self.df.iterrows():
            if len(images_dict) > image_limit:
                break
            img_name = row['media_key'].strip()
            if img_name not in images_dict.keys():
                try:
                    new_img = imageContainer(mpimg.imread(image_dir + img_name))
                    images_dict[img_name] = new_img
                except FileNotFoundError:
                    continue
        
            coord = [float(row['x']), float(row['y'])]
            polygon = json.loads(row['point_data'])['polygon']
            try:
                class_id = CLASS_NAMES.index(row['class_name'])
            except ValueError:
                class_id = -1
            images_dict[img_name].add_annot(coord, polygon, class_id)


if __name__ == "__main__":
    image_dir = './images/raw/'
    dest_dir = './images/cropped/'
    image_limit = 150

    

    for name, img in images_dict.items():
        plt.clf()
        ax = plt.gca()
        plt.imshow(img.img)
        x_coords = [point[0] for point in img.points]
        y_coords = [point[1] for point in img.points]
        plt.plot(x_coords, y_coords, 'bo')
        for polygon in img.polygons:
            ax.add_patch(polygon)

        for rect in img.rects:
            ax.add_patch(rect)
        
        plt.draw()
        plt.pause(1)
        
    
    
    # 
        # self.ax.add_collection(pc)
        # self.patch_collections.append(pc)

    
