# //////////////////////////////////////////////////////////////////////////// 
# // Copyright 2020-2021 the 3D Vision Group at the State Key Lab of CAD&CG,  
# // Zhejiang University. All Rights Reserved. 
# LICENSE in https://github.com/zju3dv/neuralbody/blob/master/LICENSE
# Slightly modified the code from the above.


import numpy as np
import cv2

from utils.geometry3d_utils import project
from config import cfg
from constant import OBJ_SEG_IDX, HAND_SEG_IDX, BOUND_SEG_IDX, LOOP_MAX_ITER


def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')

    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


# follow neuralbody style; assume we know the world bounds
def get_near_far1(bounds, ray_o, ray_d, no_before_sanitize=False):
    """calculate intersections with 3d bounding box"""
    norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
    viewdir = ray_d / norm_d
    viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
    viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
    tmin = (bounds[:1] - ray_o[:1]) / viewdir
    tmax = (bounds[1:2] - ray_o[:1]) / viewdir
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)
    near = np.max(t1, axis=-1)
    far = np.min(t2, axis=-1)
    mask_at_box = near < far

    if no_before_sanitize:
        near = near / norm_d[:, 0]
        far = far / norm_d[:, 0]
        return near, far, mask_at_box
    else:
        near = near[mask_at_box] / norm_d[mask_at_box, 0]
        far = far[mask_at_box] / norm_d[mask_at_box, 0]
        return near, far, mask_at_box

# follow pytorch3d style
def get_near_far2(ray_o, camera_center, object_center, scene_extent=8.0):
    # camera_center: (3,)
    # object_center: (3,)
    near = np.zeros_like(ray_o[:, 0], dtype=np.float32)
    far = np.zeros_like(ray_o[:, 0], dtype=np.float32)

    center_dist = np.sqrt(np.clip(np.sum(
        (camera_center - object_center) ** 2, keepdims=True), a_min=0.001, a_max=None))

    center_dist = np.clip(center_dist, a_min=scene_extent + 1e-3, a_max=None)
    min_depth = center_dist - scene_extent
    max_depth = center_dist + scene_extent

    near[:] = min_depth
    far[:] = max_depth

    return near, far, np.ones_like(near, dtype=np.bool)


# to use in building a volume feature
def sample_object_ray(msk, K, R, T, world_bounds, nrays, mode, camera_center=None, object_center=None, from_world_bounds=False, img_path=None):
    H, W = msk.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)

    coord_object = np.argwhere(msk == OBJ_SEG_IDX)
    if mode == 'train':
        nsampled_rays = 0
        ray_o_list = []
        ray_d_list = []
        near_list = []
        far_list = []
        coord_list = []

        loop_checker = 0
        while nsampled_rays < nrays:
            nobject = nrays - nsampled_rays
            # sample rays of object
            if len(coord_object) > nobject:
                coord = coord_object[np.random.randint(0, len(coord_object), nobject)]
                ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
                ray_d_ = ray_d[coord[:, 0], coord[:, 1]]

                if coord.sum() == 0:  # maybe hand mesh projection is out of the image
                    loop_checker = LOOP_MAX_ITER + 1
                    break
                else:
                    if not from_world_bounds and camera_center is not None and object_center is not None:
                        near_, far_, mask_at_box = get_near_far2(ray_o_, camera_center, object_center)
                    else:
                        near_, far_, mask_at_box = get_near_far1(world_bounds, ray_o_, ray_d_)

                    ray_o_list.append(ray_o_[mask_at_box])
                    ray_d_list.append(ray_d_[mask_at_box])
                    near_list.append(near_)
                    far_list.append(far_)
                    coord_list.append(coord[mask_at_box])
                    nsampled_rays += len(near_)

                loop_checker += 1
                if loop_checker > LOOP_MAX_ITER:
                    break

            else:
                loop_checker = LOOP_MAX_ITER + 1
                break

        if loop_checker > LOOP_MAX_ITER:
            print(f"Check the ray sampling!! img: {img_path}")
            ray_o = np.ones((1024, 3), dtype=np.float32)
            ray_d = np.ones((1024, 3), dtype=np.float32)
            near = np.zeros((1024), dtype=np.float32)
            far = np.ones((1024), dtype=np.float32)
            coord = np.zeros((1024, 2), dtype=np.int64)
        else:
            ray_o = np.concatenate(ray_o_list).astype(np.float32)
            ray_d = np.concatenate(ray_d_list).astype(np.float32)
            near = np.concatenate(near_list).astype(np.float32)
            far = np.concatenate(far_list).astype(np.float32)
            coord = np.concatenate(coord_list)

    else:  
        # nrays = nrays *3  during testing, use as much as rays
        nobject = len(coord_object)
        # sanitize
        if nobject < nrays:
            coord = coord_object
        else:
            pace = nobject // nrays
            coord = coord_object[::pace][:nrays]

        # coord = coord_object[np.random.randint(0, len(coord_object), nobject)]
        ray_o = ray_o[coord[:, 0], coord[:, 1]]
        ray_d = ray_d[coord[:, 0], coord[:, 1]]

        if not from_world_bounds and camera_center is not None and object_center is not None:
            near, far, mask_at_box = get_near_far2(ray_o, camera_center, object_center)
        else:
            near, far, mask_at_box = get_near_far1(world_bounds, ray_o, ray_d)

        near = near.astype(np.float32)
        far = far.astype(np.float32)
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]

    return ray_o, ray_d, near, far, coord

def sample_ray(img, msk, K, R, T, world_bounds, nrays, mode, render_whole=False, camera_center=None, object_center=None, from_world_bounds=False, img_path=None):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)

    # pose = np.concatenate([R, T], axis=1)
    # bound_mask = get_bound_2d_mask(world_bounds, K, pose, H, W)
    # cv2.imshow("bound_mask", bound_mask*255)
    # cv2.waitKey(0)
    # import pdb; pdb.set_trace()
    # msk = msk * bound_mask  # sgement out the mask out of the projected 3d boundary
    bound_mask = np.ones_like(img[:, :, 0], dtype=np.float32)

    if mode == 'train' and not render_whole:
        nsampled_rays = 0
        object_sample_ratio = cfg.object_sample_ratio
        hand_sample_ratio = cfg.hand_sample_ratio
        boundary_sample_ratio = cfg.boundary_sample_ratio
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        fg_mask_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []

        loop_checker = 0  # infinite loop or no coord bug
        while nsampled_rays < nrays:
            n_object = int((nrays - nsampled_rays) * object_sample_ratio)
            n_hand = int((nrays - nsampled_rays) * hand_sample_ratio)
            n_boundary = int((nrays - nsampled_rays) * boundary_sample_ratio)
            n_rand = (nrays - nsampled_rays) - n_object - n_hand - n_boundary
            coord_object = np.argwhere(msk == OBJ_SEG_IDX)
            coord_hand = np.argwhere(msk == HAND_SEG_IDX)
            coord_boundary = np.argwhere(msk == BOUND_SEG_IDX)

            # sample rays on object
            if len(coord_object) > 0:
                coord_object = coord_object[np.random.randint(0, len(coord_object), n_object)]
            # sample rays on hand
            if len(coord_hand) > 0:
                coord_hand = coord_hand[np.random.randint(0, len(coord_hand), n_hand)]
            # sample rays on boundary of hand+object mask
            if len(coord_boundary) > 0:
                coord_boundary = coord_boundary[np.random.randint(0, len(coord_boundary), n_boundary)]

            coord_rand = np.argwhere(bound_mask == 1)
            if len(coord_object) == 0 and len(coord_hand) == 0 and len(coord_boundary):
                # target object or hand is not visible in this view
                if len(coord_rand) < (nrays - nsampled_rays):
                    loop_checker = LOOP_MAX_ITER + 1
                    break
                else:
                    coord_rand = coord_rand[np.random.randint(0, len(coord_rand), nrays - nsampled_rays)]
            else:
                coord_rand = coord_rand[np.random.randint(0, len(coord_rand), n_rand)]

            coord = np.concatenate(
                [coord_object, coord_hand, coord_boundary, coord_rand], axis=0)

            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]
            fg_mask_ = msk[coord[:, 0], coord[:, 1]]

            if coord.sum() == 0:  # maybe hand mesh projection is out of the image
                loop_checker = LOOP_MAX_ITER + 1
                break
            else:
                if not from_world_bounds and camera_center is not None and object_center is not None:
                    near_, far_, mask_at_box = get_near_far2(ray_o_, camera_center, object_center)
                else: # from_world_bounds
                    near_, far_, mask_at_box = get_near_far1(world_bounds, ray_o_, ray_d_)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            fg_mask_list.append(fg_mask_[mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            coord_list.append(coord[mask_at_box])
            mask_at_box_list.append(mask_at_box[mask_at_box])
            nsampled_rays += len(near_)

            loop_checker += 1
            if loop_checker > LOOP_MAX_ITER:
                break

        if loop_checker > LOOP_MAX_ITER:    
            print(f"Check the ray sampling!! img: {img_path}")
            ray_o = np.ones((1024, 3), dtype=np.float32)
            ray_d = np.ones((1024, 3), dtype=np.float32)
            rgb = np.ones((1024, 3), dtype=np.float32)
            fg_mask = np.zeros((1024), dtype=np.float32)
            near = np.zeros((1024), dtype=np.float32)
            far = np.ones((1024), dtype=np.float32)
            coord = np.zeros((1024, 2), dtype=np.int64)
            mask_at_box = np.zeros((1024), dtype=bool)
        else:
            ray_o = np.concatenate(ray_o_list).astype(np.float32)
            ray_d = np.concatenate(ray_d_list).astype(np.float32)
            rgb = np.concatenate(rgb_list).astype(np.float32)
            fg_mask = np.concatenate(fg_mask_list).astype(np.float32)
            near = np.concatenate(near_list).astype(np.float32)
            far = np.concatenate(far_list).astype(np.float32)
            coord = np.concatenate(coord_list)
            mask_at_box = np.concatenate(mask_at_box_list)

    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        fg_mask = msk.reshape(-1).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)

        if not from_world_bounds and camera_center is not None and object_center is not None:
            near, far, mask_at_box = get_near_far2(ray_o, camera_center, object_center)

        else:  # from_world_bounds
            # During training, it is very inefficient to have varying size of data in datalaoder; make sure no_before_sanitize=True
            no_before_sanitize = (mode == 'train' and render_whole)
            near, far, mask_at_box = get_near_far1(world_bounds, ray_o, ray_d, no_before_sanitize)

            # if near and far are already sanitized and thus len(near) < cfg.N_rand, match rgb, ... to near and far
            if not no_before_sanitize: # if mode == 'test', do this.
                rgb = rgb[mask_at_box]
                fg_mask = fg_mask[mask_at_box]
                ray_o = ray_o[mask_at_box]
                ray_d = ray_d[mask_at_box]

        near = near.astype(np.float32)
        far = far.astype(np.float32)
        coord = np.zeros([len(rgb), 2]).astype(np.int64)

    return rgb, fg_mask, ray_o, ray_d, near, far, coord, mask_at_box
