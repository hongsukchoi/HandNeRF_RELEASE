# obtain maskrcnn prediction from a dexycb cvpur result file, decode save it agin
import json
import cv2
import numpy as np
import os.path as osp
import pycocotools.mask
import time

from collections import defaultdict
from tqdm import tqdm

# For reference  of category ids; https://github.com/NVlabs/dex-ycb-toolkit/blob/master/dex_ycb_toolkit/dex_ycb.py
_YCB_CLASSES = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}

if __name__ == '__main__':
    data_dir = '/home/hongsuk.c/Projects/HandNeRF/data/DexYCB/data'
    annot_path = '/home/hongsuk.c/Projects/HandNeRF/data/DexYCB/data/annotation/s0_test.json'
    dexycb_cvpr_result_file = '/home/hongsuk.c/Projects/dex-ycb-toolkit/results/cvpr2021_results/coco_maskrcnn_s0_test.json'
    target_img_list_path = '/home/hongsuk.c/Projects/HandNeRF/data/DexYCB/data/annotation/novel_object_test_list.json'  #TEMP
    
    file_read_start = time.time()
    with open(annot_path, 'r') as f:
        annot_list = json.load(f)
    
    with open(dexycb_cvpr_result_file, 'r') as f:
        dexycb_cvpr_maskrcnn_result = json.load(f)

    # with open(target_img_list_path, 'r') as f:
    #     target_img_list_dict =json.load(f)
    # target_img_idx_list = target_img_list_dict.keys()
    file_read_end = time.time()
    print("File read time... ", file_read_end - file_read_start)
    target_img_idx_list = [1,2,3,4,5,6,30,60]

    extracted_maskrcnn_result = defaultdict(list)

    for maskrcnn_result in dexycb_cvpr_maskrcnn_result:
        idx = maskrcnn_result['image_id']
        if idx not in target_img_idx_list:
            continue

        category_id = maskrcnn_result['category_id']
        rle = maskrcnn_result['segmentation']
        seg = pycocotools.mask.decode(rle) # (480,640), uint8

        annot = annot_list[idx]
        rgb_path = annot['color_file']  # RGB image '/home/hongsuk/Data/DexYCB/20200820-subject-03/20200820_135508/836212060125/color_000000.jpg'
        depth_path = annot['depth_file']  # 3 channel depth image 0~255 '/home/hongsuk/Data/DexYCB/20200709-subject-01/20200709_142123/836212060125/aligned_depth_to_color_000000.png'
        label_path = annot['label_file']  # '/home/hongsuk/Data/DexYCB/20200820-subject-03/20200820_135508/836212060125/label_000000.npz'
        # Replace the absolute directory
        # https://github.com/NVlabs/dex-ycb-toolkit
        rgb_path = rgb_path.replace('/home/hongsuk/Data/DexYCB', data_dir)
        depth_path = depth_path.replace('/home/hongsuk/Data/DexYCB', data_dir)
        label_path = label_path.replace('/home/hongsuk/Data/DexYCB', data_dir)

        # https://github.com/NVlabs/dex-ycb-toolkit/tree/master
        if category_id != 22: # if it's object
            seg = seg * category_id
        else: # hand. follow the dexycb seg annot format
            seg = seg * 255
        
        # rgb = cv2.imread(rgb_path)
        # rgb[seg == 1] = 0
        # cv2.imshow('seg vis', rgb)
        # cv2.waitKey(0)       
        extracted_maskrcnn_result[idx].append(seg) 

    for idx in tqdm(extracted_maskrcnn_result.keys()):
        maskrcnn_seg = sum(extracted_maskrcnn_result[idx])
        rgb_path = annot_list[idx]['color_file'].replace('/home/hongsuk/Data/DexYCB', data_dir)

        maskrcnn_seg_path = '/' + osp.join(*rgb_path.split('/')[:-1], osp.basename(rgb_path).replace('color', 'maskrcnn_seg').replace('jpg', 'npy'))
        np.save(maskrcnn_seg_path, maskrcnn_seg)
    
