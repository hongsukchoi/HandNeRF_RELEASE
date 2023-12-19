# Custom data collection

One of the main contributions of HandNeRF is that it can be trained on casually collected data. Here, "casual" means that the collection process is cheaper and easier in general thatn 3D data colleciton.

## Image capture

## Camera calibration

## Bounding box detection


## 3D hand estimation

### 2D pose estimation


### Fitting MANO parameters

Add this to L1683 of `envs/handnerf/lib/python3.7/site-packages/smplx/body_models.py`.   
```     
if self.is_rhand:
    tips = vertices[:, [745, 317, 444, 556, 673]]
else:
    tips = vertices[:, [745, 317, 445, 556, 673]]
    joints = torch.cat([joints, tips], dim=1)
```


## Object segmentation


