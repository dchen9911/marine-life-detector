from show_images import imageSet
from PIL import Image

def crop_random(img_dir, min_size=100):
    img_set = imageSet(image_dir, size_limit=500)
    for name, img in img_set.images_list:
        im_height, im_width, _ = img.shape
        # WIP

    img_arr = img.img
    pil_img = Image.fromarray(img_arr)
    pil_img = Image.fromarray(img_arr)


# image_set = imageSet(image_dir)
# images_dict = image_set.images_dict
# name_dict = {}
 
# dest_dir = './images/cropped/'
# # saving the cropped images
# for name, img in images_dict.items():
#     for class_id, bbox in zip(img.class_ids, img.bboxes):
#         img_arr = img.img
#         pil_img = Image.fromarray(np.uint8(img_arr*255))
#         x0, y0, width, height = [int(i) for i in bbox.bounds]
#         crop_area = (x0, y0, x0 + width, y0 + height)
#         cropped_img = pil_img.crop(crop_area)

#         if class_id == -1:
#             name = 'other'
#         elif class_id == 2: # for foliose
#             name = 'foliose'
#         else:
#             name = CLASS_NAMES[class_id]
            
        
#         if name not in name_dict.keys():
#             name_dict[name] = 1
#         else:
#             name_dict[name] += 1
#         name_str = dest_dir + name + '_' + str(name_dict[name]) + '.png'
#         cropped_img.save(name_str)