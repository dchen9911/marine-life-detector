from consts import *
import shutil 
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import json
from matplotlib.patches import Polygon, Rectangle



info_path = base_path + 'rcnn_log.csv'
file_dir = base_path + 'images/negative_raw_test/'
df = pd.read_csv(info_path)


for barrier1 in [0.6]:
    img_dest = base_path + 'images/negative_raw_test/' + str(int(barrier1*100)) + ' bigthreshold/'
    if not os.path.isdir(img_dest):
        os.mkdir(img_dest)

    for ind, row in df.iterrows():
        img_name = row['name']
        img_path = file_dir + img_name
        if not os.path.isfile(img_path):
            continue

        results_dict = json.loads(row['results'])
        dims = results_dict['dims']
        probs = results_dict['probs']
        min_probs = results_dict['min_probs']
        max_probs = results_dict['max_probs']
        started_plot = False
        for dim, prob, min_prob, max_prob in zip(dims, probs, min_probs, max_probs):
            x, y, w, h = dim
            barrier2 = 0.992
            barrier3 = 0.995
            
            max_barrier = 0.8
            if w < 120 and h < 120:
                min_barrier = barrier3
            elif w < 120 or h < 120:
                min_barrier = barrier2
            elif w > 500 or h > 500:
                min_barrier = barrier3
            elif w > 400 or h > 400:
                min_barrier = barrier2
            else:
                # max_barrier = 0.99
                min_barrier = barrier1

            if min_prob < 0.7:
                continue        
            if prob > 0.8 and (min_prob > min_barrier and max_prob > max_barrier):
                if not started_plot:
                    plt.figure(figsize=(18,18))
                    ax = plt.gca()
                    image = cv2.imread(img_path)[...,::-1]
                    ax.imshow(image)
                    started_plot = True
                rect = Rectangle([x ,y], w, h, fill=False, color='red', lw=3.5)
                ax.add_patch(rect)
                prob_str = str(min_prob)[0:6]   
                # str(prob)[0:6] 
                # prob_str += ', ' + 
                # prob_str += ', ' + str(max_prob)[0:6]
                ax.text(x + 2,y + 3, prob_str, size='large', c='red', ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, lw=0))
        
        if started_plot:
            save_file = img_dest + img_name.split('.')[0] + '(1)' + '.png'
            plt.savefig(save_file, bbox_inches='tight')
            plt.close()
        print(ind, end=',', flush=True)

