# Data

You need to follow directory structure of the data as below.

```
${ROOT}  
|-- data 
|   |-- DexYCB
|   |   |-- data
|   |   |-- DexYCB.py
|   |-- HO3D
|   |   |-- data
|   |   |-- HO3D.py
|   |-- Custom
|   |   |-- data
|   |   |-- Custom.py
```

- `data` is the directory to each dataset's data.
- We recommend you to use a softlink, since the storage will be huge.


## DexYCB

- Download DeXYCB images and other data from [here](https://dex-ycb.github.io/).
- Download all annotation files from [here](https://drive.google.com/drive/folders/17ZCvAtVqMo_DBJ7CBKCi1vxMSoGGmXha?usp=sharing).

Or create annotation files yourself

- Generate annnotation files (e.g., `s0_test.json`) following [here](https://github.com/NVlabs/dex-ycb-toolkit#loading-dataset-and-visualizing-samples).
- Go to the `tool` folder and run `python3 save_dexycb_world_data.py --dexycb_dir /path/to/dexycb`


The direcotry should look like below

```
${/path/to/dexycb}
|-- annotation  # we use s0 set
|   |-- s0_train.json
|   |-- s0_train_world_data.json
|   |-- s0_val.json
|   |-- s0_val_world_data.json
|   |-- s0_test.json
|   |-- s0_test_world_data.json 
# Refer to 'Prepare HandOccNet ~' 
|   |-- DEX_YCB_s0_val_data.json # annotation file to run HandOccNet
|   |-- DEX_YCB_s0_test_data.json # annotation file to run HandOccNet
|   |-- novel_object_test_list.json # annotation file to run HandOccNet
|   |-- novel_object_grasp_list.json # annotation file to run HandOccNet
|   |-- DexYCB_HandNeRF_novel_object_testset_HandOccNet_pred.npy # input for HandNeRF
|   |-- DexYCB_HandNeRF_novel_grasp_testset_HandOccNet_pred.npy  # input for HandNeRF
#
|-- models
|-- calibration
|-- bop
|-- 20200709-subject-01 
|-- 20200813-subject-02
|-- 20200820-subject-03
|-- 20200903-subject-04 
|-- 20200908-subject-05
|-- 20200918-subject-06
|-- 20200928-subject-07
|-- 20201002-subject-08
|-- 20201015-subject-09
|-- 20201022-subject-10

```

## HO3D (v3)

- Download HO3D v3 images and other data from [here](https://cloud.tugraz.at/index.php/s/z8SCsWCYM3YcQWX?)
- Download all annotation from [here](https://drive.google.com/drive/folders/1rHPSXpubKvmbpZTEKlPPjxtPLQG051HE?usp=sharing)

Or create annotation files yourself

- Go to the `tool` folder and run `python3 ho3d2coco.py --ho3d_dir /path/to/ho3d --dexycb_dir /path/to/dexycb`
- `dexycb_dir` is used to refer to object model files.

The direcotry should look like below

```
${/path/to/ho3d}
|-- annotations
|   |-- HO3Dv3_partial_train_multiseq_coco.json
|   |-- HO3Dv3_partial_train_multiseq_world_data.json
|   |-- HO3Dv3_partial_test_multiseq_coco.json
|   |-- HO3Dv3_partial_test_multiseq_world_data.json
# Refer to 'Prepare HandOccNet ~' 
|   |-- HO3Dv3_eval_for_handoccnet.json # annotation file to run HandOccNet
|   |-- novel_grasp_test_list.json # annotation file to run HandOccNet
|   |-- HO3Dv3_HandNeRF_novel_grasp_testset_HandOccNet_pred.npy  # input for HandNeRF
#
|-- calibration
|-- manual_annotations
|-- joint_order.png
|-- README.txt
|-- evaluation  # image files
|-- train # image files
|-- evaluation.txt
|-- train.txt
```


## Prepare HandOccNet test input for DexYCB and HO3D v3

- Go to the `HandOccNet` directory.
- Download the HandOccNet pretrained on the DexYCB s0 trainset from [here](https://drive.google.com/drive/folders/1R3kkJ8NQDpGDGNKDbLDXUBim0B1SG5WV) and place it under `./HandOccNet/output/model_dump/`.
- For DexYCB, go to the `tool` folder and run 
```
python3 dexycb2coco_handoccnet.py --dexycb_dir /path/to/dexycb
``` 
to get annotation files in the HandOccnet format. Place the produced `DEX_YCB_s0_test_data.json` under `/path/to/DexYCB/annoation`.
- For HO3D v3, go to the `tool` folder and run 
```
python3 ho3d2coco_handoccnet.py --ho3d_dir /path/to/ho3d
```
to get annotation files in the HandOccnet format. Place the produced `HO3Dv3_eval_for_handoccnet.json` under `/path/to/HO3D/annoations`.
- Make softlinks to `/path/to/DexYCB` and `/path/to/HO3D` as `./HandOccNet/data/DEX_YCB/data` and `./HandOccNet/data/HO3D/data` respectively.
- Go to the `./HandOccNet/main` directory and set `testset` to 'DEX_YCB' or 'HO3D' and set `test_config` to 'novel_object' (only for DexYCB) or 'novel_grasp'.
- Run 
```
python3 save_handnerf_input.py --gpu 0-3 --test_epoch 25
```
- Place the result file in `/path/to/DexYCB/annoation` or `/path/to/HO3D/annoations`


 
## Custom dataset

This data is a small dataset to elaborate how to run HandNeRF on users' custom data. Refer to [here](./custom_data_collection.md) to learn how to relatively casually collect data that can supervise HandNeRF.

- [Waiting for permission]~~Download images and annotation files from [here](https://drive.google.com/drive/folders/1gmeTQL8qRVhctGqIxiEMc03Mttq8TGCp?usp=sharing).~~

Or create annotation files yourself

- Go to the `tool` folder and run `python custom2coco.py --custom_dir /path/to/custom`


The direcotry should look like below

```
${/path/to/ho3d}
|-- cam_params_final.json  # intrinsinc & extrinsic camera data
|-- custom_train_data.json # anotation for running HandNeRF
|-- custom_test_data.json # anotation for running HandNeRF 
|-- train # contains image, handoccnet mesh, OpenPose 2D pose, SAM segmentation
|   |-- cam_0 # RGB images
|   |-- cam_0_handoccnet # handoccnet estimation on a single image
|   |-- cam_0_keypoints # OpenPose 2D pose estimation
|   |-- cam_0_segmentation # SAM (Segment-Anything) segmentation
|   |-- cam_1 
|   |-- .
|   |-- .
|   |-- .
|   |-- cam_6_segmentation
|-- test
```
