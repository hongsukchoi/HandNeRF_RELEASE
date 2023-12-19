# For HandOccNet
import cv2
import numpy as np
import os.path as osp
import json
import argparse

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dexycb_dir', type=str, default='')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    root_path = args.dexycb_dir # '/data/hongsuk/data/DexYCB'
    category_list = ['s0'] #, 's1', 's2', 's3']
    set_list = [ 'val', 'test']
    subsample_ratio = 1

    img_id = 0; annot_id = 0
    images = []; annotations = []
    for category_split in category_list:
        for set_split in set_list:
            world_data = {}
            json_path = osp.join(root_path, 'annotation', f'{category_split}_{set_split}.json')

            with open(json_path, 'r') as f:
                annot_list = json.load(f)

            print(f"Parsing world data of split {category_split} {set_split} set ...")
            
            for annot in tqdm(annot_list):
                label_path = annot['label_file']
                label_path = label_path.replace('/home/hongsuk/Data/DexYCB', root_path)

                img_path = label_path.replace('labels', 'color')[:-3] + 'jpg'
                img = cv2.imread(img_path)
                height, width = img.shape[:2]
                file_name = '/'.join(img_path.split('/')[-4:])

                hand_type = annot['mano_side']

                img_dict = {}
                annot_dict = {}

                img_dict['width'] = width
                img_dict['height'] = height
                img_dict['file_name'] = file_name
                img_dict['id'] = img_id

                annot_dict['id'] = annot_id
                annot_dict['image_id'] = img_id
                annot_dict['joints_coord_cam'] = np.zeros((21,3)).tolist()
                annot_dict['joints_img'] = np.zeros((21,2)).tolist()
                annot_dict['hand_type'] = hand_type
                annot_dict['cam_param'] = {}

                img_id += 1
                annot_id += 1

                images.append(img_dict)
                annotations.append(annot_dict)

            output = {'images': images, 'annotations': annotations}
            output_path = osp.join(root_path, f'DEX_YCB_{category_split}_{set_split}_data.json')
            with open(output_path, 'w') as f:
                json.dump(output, f)
            print('Saved at ' + output_path)