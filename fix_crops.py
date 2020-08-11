import pandas as pd
import json
from consts import *
df = pd.read_csv(base_path + 'crop_info_2.csv')

out_file = open(base_path + 'crop_info_fixed.csv', 'w+')

for ind, row in df.iterrows():            
    img_name = row['name'].strip()
    crop_areas = json.loads(row['area'])['crop_areas']
    correct_areas = []
    for crop_area in crop_areas:
        if len(correct_areas) == 0:
            correct_areas.append(crop_area)
        else:
            if crop_area == correct_areas[-1]:
                continue
            else:
                correct_areas.append(crop_area)
    new_crop_dict = {'crop_areas':correct_areas}
    out_file.write(img_name + ',' + '\"' + 
                                      json.dumps(new_crop_dict) + '\"' + '\n') 

