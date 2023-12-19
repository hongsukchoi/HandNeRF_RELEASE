# for preprocessing the 2d mask
OBJ_SEG_IDX = 123
HAND_SEG_IDX = 234
BOUND_SEG_IDX = 345

# for semantic nerf supervision and marching cube during 3D mesh reconstruction
OBJ_SEMANTIC_IDX = 1
HAND_SEMANTIC_IDX = 2
MAX_NUM_CLASSES = 100

# Mano specific
ROOT_JOINT_IDX = 0  # wrist; https://github.com/NVlabs/dex-ycb-toolkit/blob/master/dex_ycb_toolkit/dex_ycb.py#L59-81

# DEX-YCB specific
ANNOT_HAND_SEG_IDX = 255  # https://github.com/NVlabs/dex-ycb-toolkit#loading-dataset-and-visualizing-samples
DEXYCB_OBJECT_CLASSES = {
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

LOOP_MAX_ITER = 10

# CLIP arguments
CLIP_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
import torchvision
# normalize an image that is already scaled to [0, 1]
CLIP_NORMALIZE = torchvision.transforms.Normalize(CLIP_MEAN, CLIP_STD)

# for 3d iou
# reorder the corners from open3d definition to objectron box
open3d2objectron = [0, 1, 2, 4, 7, 3, 8, 6, 5]
