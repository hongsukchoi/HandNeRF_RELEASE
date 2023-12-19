import cv2
import open3d as o3d
import numpy as np
import torch

from PIL import Image
from typing import List, Dict

from constant import ANNOT_HAND_SEG_IDX, OBJ_SEG_IDX, HAND_SEG_IDX, open3d2objectron
from utils.bbox3d.box import Box
from utils.bbox3d.iou import IoU
from utils.preprocessing import load_img

def get_pointcloud_from_rgbd(rgb_path: str, label_path: str, depth_path: str, K: np.ndarray, grabbing_object_id: int, dataset='DexYCB'):
    # return pointcloud of xyz, color, semantic label
    if dataset == 'DexYCB':
        img = load_img(rgb_path) / 255.  # 0~1.
        depth = np.asarray(Image.open(depth_path)) / 1000.  # mm -> m

        seg = np.load(label_path)['seg']
        seg[(seg != ANNOT_HAND_SEG_IDX) & (seg != grabbing_object_id)] = 0
        j, i = (seg != 0).nonzero() 

        xy1 = np.stack([i, j, np.ones_like(i)], axis=1)
        imgxy_and_z = xy1 * depth[seg != 0][:, None]

        cam_pointcloud = np.dot(imgxy_and_z, np.linalg.inv(K).T)
        
        # vis
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(cam_pointcloud)
        # pcd.colors = o3d.utility.Vector3dVector(img[seg != 0])
        # o3d.io.write_point_cloud("check.ply", pcd)

        xyz_color_label = np.concatenate([cam_pointcloud, img[seg != 0], seg[seg != 0][:, None]], axis=1)

        return xyz_color_label
    else:
        raise NotImplementedError("Not implemented yes for pointcloud from rgbd")
    

# return center + corners of oriented bounding box from a point cloud
def get_oriented_3d_bbox(point_cloud: np.ndarray) -> np.ndarray:
    """
    point_cloud: (N, 3) np.ndarray

    Return
    vertices: (9, 3); 1 center + 8 corners
    
    """
    o3d_points = o3d.utility.Vector3dVector(point_cloud)
    oriented_3d_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d_points)
    center = oriented_3d_bbox.get_center()  # (3,)
    corners = oriented_3d_bbox.get_box_points()  # (8, 3)

    vertices = np.concatenate([np.asarray(center)[None, :], np.asarray(corners)])
    # reorder
    vertices = vertices[open3d2objectron]

    return vertices


# return 3d bounding box iou of two meshes (pointclouds)
def calc_3d_iou(mesh_a: np.ndarray, mesh_b: np.ndarray):
    """"
    mesh_a: (N1, 3)
    mesh_b: (N2, 3)

    Return
    iou: scalar 0~1
    
    """

    bbox_points_a = get_oriented_3d_bbox(mesh_a)  # (9, 3) center + 8 corners
    bbox_points_b = get_oriented_3d_bbox(mesh_b)  # (9, 3) center + 8 corners

    objectron_box_a = Box(bbox_points_a) 
    objectron_box_b = Box(bbox_points_b)
    
    IoU_calculator = IoU(objectron_box_a, objectron_box_b)
    iou = IoU_calculator.iou()

    return iou

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


# batch tensor computation for getting surface function; ax + by + cz + d = 0
# Refer to: https://kitchingroup.cheme.cmu.edu/blog/2015/01/18/Equation-of-a-plane-through-three-points/
def tensor_get_surfaces(faces):
    """
    faces: (N, 3, 3), N surfaces of 3 points with xyz
    
    // Returns //
    surfaces : (N, 4), a, b, c, d
    """

    # two vectors in the plane
    v1 = faces[:, 2, :] - faces[:, 0, :]
    v2 = faces[:, 1, :] - faces[:, 0, :]

    surface_normal_vector = torch.cross(v1, v2, dim=1)  # batch computation
    # surface_normal: (N, 3), a, b, c
    surface_norm = torch.norm(surface_normal_vector, dim=1, p='fro')
    surface_normal_vector = surface_normal_vector / surface_norm[:, None]
    d = (surface_normal_vector * faces[:, 2, :]).sum(dim=1)  # (N, )
    d = -d

    surfaces = torch.cat([surface_normal_vector, d[:, None]], dim=1)
    
    return surfaces

# batch tensor computation to get projection distance and foot
def tensor_get_projection(points, surfaces):
    """
    Distance = (| a*x1 + b*y1 + c*z1 + d |) / (sqrt( a*a + b*b + c*c))

    points: (N, 3)
    surfaces: (1538, 4), a, b, c, d
    
    // Returns // 
    distances: (N, 1538)
    foot: (N,1538,3)
    """
    points_homo = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
    numerator = torch.inner(points_homo, surfaces)  # (N, 1538); a*x1 + b*y1 + c*z1 + d
    denominator = (surfaces[:, :3] ** 2).sum(dim=1)  # (1538, ); a*a + b*b + c*c
    distances = torch.abs(numerator) / torch.sqrt(denominator[None, :])  # (N, 1538)

    direction = - numerator / denominator[None, :]  # (N, 1538); (-a * x1 - b * y1 - c * z1 - d) / (a * a + b * b + c * c)
    feet = points[:, None, :] + surfaces[None, :, :3] * direction[:, :, None]  # (N, 1538, 3)

    return distances, feet

# batch tensor computation for topk
def tensor_get_shortest_distance_from_surfaces(points, surfaces, k=1):
    """
    Distance = (| a*x1 + b*y1 + c*z1 + d |) / (sqrt( a*a + b*b + c*c))

    points: (N, 3)
    surfaces: (1538, 4), a, b, c, d
    
    // Returns //
    value: (N, k), min value
    indices: (N, k), indices of min value in surfaces
    """

    distances, feet = tensor_get_projection(points, surfaces)

    # value, indices = distances.min(dim=1)
    values, indices = torch.topk(-distances, k, dim=-1)  # (N, k), (N, k)
    # inverse the minus distance to positive
    values = -values

    return values, indices

# get the inside points from 2D target object masks //  for DexYCB data 3D reconstruction using marching cube
# if grabbing_objet_id != -1, DexYCB data
# else HanCo data
def prepare_inside_world_pts(wpts: np.ndarray, R_list: List[np.ndarray], t_list: List[np.ndarray], K_list: List[np.ndarray], label_path_list: List[str], grabbing_object_id: int, dataset: str) -> np.ndarray:
    """
    wpts: (w, h, d 3), points to reconstruct
    R_list: list of rotation matrix (3,3)
    t_list: list of translation matrix (3,1)
    K_list: list of intrinsics matrix (3,3)
    label_path_list: list of label path

    // Returns //
    inside: (w, h, d)
    """
    sh = wpts.shape
    wpts3d = wpts.reshape(-1, 3)

    inside = np.ones([len(wpts3d)]).astype(np.uint8)

    for R, t, K, label_path in zip(R_list, t_list, K_list, label_path_list):
        ind = inside == 1
        wpts3d_ = wpts3d[ind]

        RT = np.concatenate([R, t], axis=1)
        pts2d = project(wpts3d_, K, RT)

        if dataset == 'DexYCB':  # parse mask as DexYCB
            # np.uint8, (H,W) // 0~255
            seg = np.load(label_path)['seg']
            mask = seg.copy()            
            mask[(seg != ANNOT_HAND_SEG_IDX) & (seg != grabbing_object_id)] = 0
            mask[mask != 0] = 1
        elif dataset == 'Custom':
            # tuple, bool
            hand_seg = np.load(label_path[0], allow_pickle=True)[()]['seg']
            obj_seg = np.load(label_path[1], allow_pickle=True)[()]['seg']
            mask = (hand_seg | obj_seg).astype(int)

        else: # HO3D, HanCo
            mask = cv2.imread(label_path)[:, :, :]  # np.uint8
            mask = mask.sum(axis=-1)
            mask[mask != 0] = 1
            if dataset == 'HO3D':
                mask = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_NEAREST)
        # else: # parse mask as HanCo
        #     mask = cv2.imread(label_path)[:, :, 0]  # np.uint8
        #     mask[mask != 0] = 1

        # Skip partial views where full object is not shown
        if mask.sum() < 1000:
            continue

        H, W = mask.shape
        pts2d = np.round(pts2d).astype(np.int32)
        pts2d[:, 0] = np.clip(pts2d[:, 0], 0, W - 1)
        pts2d[:, 1] = np.clip(pts2d[:, 1], 0, H - 1)
        mask_ = mask[pts2d[:, 1], pts2d[:, 0]]

        inside[ind] = mask_

    inside = inside.reshape(*sh[:-1])

    return inside


# get the inside points from 3D intersection of 2D target object mask rays //  for sampling object points in 3D space
def prepare_inside_world_object_pts(wpts: np.ndarray, R_list: List[np.ndarray], t_list: List[np.ndarray], K_list: List[np.ndarray], mask_list: List[np.ndarray]) -> np.ndarray:
    """
    wpts: (w, h, d 3), points to reconstruct
    R_list: list of rotation matrix (3,3)
    t_list: list of translation matrix (3,1)
    K_list: list of intrinsics matrix (3,3)
    mask_list: list of masks (h,w)

    // Returns //
    inside: (w, h, d)
    """
    sh = wpts.shape
    wpts3d = wpts.reshape(-1, 3)

    inside = np.ones([len(wpts3d)]).astype(np.uint8)
    for R, t, K, mask in zip(R_list, t_list, K_list, mask_list):
        ind = inside == 1
        wpts3d_ = wpts3d[ind]

        RT = np.concatenate([R, t], axis=1)
        pts2d = project(wpts3d_, K, RT)

        # np.uint8, (H,W) // 0~255
        mask[mask != OBJ_SEG_IDX] = 0
        mask[mask == OBJ_SEG_IDX] = 1
        # Skip partial views where full object is not shown
        if mask.sum() < 1000:
            continue

        H, W = mask.shape
        pts2d = np.round(pts2d).astype(np.int32)
        pts2d[:, 0] = np.clip(pts2d[:, 0], 0, W - 1)
        pts2d[:, 1] = np.clip(pts2d[:, 1], 0, H - 1)
        mask_ = mask[pts2d[:, 1], pts2d[:, 0]]

        inside[ind] = mask_

    inside = inside.reshape(*sh[:-1])

    return inside

# get the inside points from 3D intersection of 2D target object mask rays //  for sampling object points in 3D space
def torch_prepare_inside_world_object_pts(wpts: torch.Tensor, R: torch.Tensor, t: torch.Tensor, K: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    wpts: (1, w, h, d 3), points to reconstruct
    R: rotation matrix (1,3,3)
    t: translation matrix (1,3,1)
    K: intrinsics matrix (1,3,3)
    mask: binary mask (1,h,w)

    // Returns //
    inside: (1, w, h, d)
    """
    sh = wpts.shape
    wpts3d = wpts.reshape(1,-1, 3)

    # project world points
    cpts3d = torch.bmm(R, wpts3d.transpose(1,2)).transpose(1,2) + t.transpose(1,2)  # (1, -1, 3)
    homopts3d = torch.bmm(cpts3d, K.transpose(1,2))
    imgpts2d = homopts3d[:, :, :2]/ homopts3d[:, :, 2:]

    # sample mask pixel value of the points
    _, H, W = mask.shape
    pts2d = imgpts2d.round()
    pts2d[:, :, 0] = torch.clamp(pts2d[:, :, 0], 0, W - 1)
    pts2d[:, :, 1] = torch.clamp(pts2d[:, :, 1], 0, H - 1)
    pts2d = pts2d.long()
    inside = mask[0, pts2d[0, :, 1], pts2d[0, :, 0]]
    inside= inside.reshape(*sh[:-1])
    return inside

# get the inside points from 3D intersection of 2D target object mask rays //  for sampling object points in 3D space
def prepare_inside_world_hand_pts(wpts: np.ndarray, R_list: List[np.ndarray], t_list: List[np.ndarray], K_list: List[np.ndarray], mask_list: List[np.ndarray]) -> np.ndarray:
    """
    wpts: (w, h, d 3), points to reconstruct
    R_list: list of rotation matrix (3,3)
    t_list: list of translation matrix (3,1)
    K_list: list of intrinsics matrix (3,3)
    mask_list: list of masks (h,w)

    // Returns //
    inside: (w, h, d)
    """
    sh = wpts.shape
    wpts3d = wpts.reshape(-1, 3)

    inside = np.ones([len(wpts3d)]).astype(np.uint8)
    for R, t, K, mask in zip(R_list, t_list, K_list, mask_list):
        ind = inside == 1
        wpts3d_ = wpts3d[ind]

        RT = np.concatenate([R, t], axis=1)
        pts2d = project(wpts3d_, K, RT)

        # np.uint8, (H,W) // 0~255
        mask[mask != HAND_SEG_IDX] = 0
        mask[mask == HAND_SEG_IDX] = 1
        # Skip partial views where full object is not shown
        if mask.sum() < 1000:
            continue

        H, W = mask.shape
        pts2d = np.round(pts2d).astype(np.int32)
        pts2d[:, 0] = np.clip(pts2d[:, 0], 0, W - 1)
        pts2d[:, 1] = np.clip(pts2d[:, 1], 0, H - 1)
        mask_ = mask[pts2d[:, 1], pts2d[:, 0]]

        inside[ind] = mask_

    inside = inside.reshape(*sh[:-1])

    return inside

# get 3D grid points from 3D bounds
def get_grid_points_from_bounds(bounds: np.ndarray, voxel_size: List[float], mode: str) -> np.ndarray:
    """
    bounds: (2,3)
    voxel_size: dhw
    mode: train or test

    // Returns // 
    grid_points: (x_size, y_size, z_size, 3) xyz
    """
    x = np.arange(bounds[0, 0], bounds[1, 0] + voxel_size[2], voxel_size[2])
    y = np.arange(bounds[0, 1], bounds[1, 1] + voxel_size[1], voxel_size[1])
    z = np.arange(bounds[0, 2], bounds[1, 2] + voxel_size[0], voxel_size[0])
    grid_points = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).astype(np.float32)

    # put randomness to the grid points for data augmentation perspective
    if mode == 'train':
        grid_points = grid_points + np.random.random(grid_points.shape).astype(np.float32) * np.array(voxel_size, dtype=np.float32)[[2,1,0]]
        # sanitize
        grid_points[:, :, :, 0] = np.clip(grid_points[:, :, :, 0], a_min=bounds[0,0], a_max=bounds[1,0])
        grid_points[:, :, :, 1] = np.clip(grid_points[:, :, :, 1], a_min=bounds[0,1], a_max=bounds[1,1])
        grid_points[:, :, :, 2] = np.clip(grid_points[:, :, :, 2], a_min=bounds[0,2], a_max=bounds[1,2])

    return grid_points

def torch_get_grid_points_from_bounds(bounds: torch.Tensor, voxel_size: List[float], mode: str) -> torch.Tensor:
    """
    bounds: (1,2,3)
    voxel_size: dhw
    mode: train or else

    // Returns // 
    grid_points: (1, x_size, y_size, z_size, 3) xyz
    """
    x = torch.arange(bounds[0, 0, 0], bounds[0,1, 0] + voxel_size[2], voxel_size[2], device=bounds.device)
    y = torch.arange(bounds[0, 0, 1], bounds[0, 1, 1] + voxel_size[1], voxel_size[1], device=bounds.device)
    z = torch.arange(bounds[0, 0, 2], bounds[0, 1, 2] + voxel_size[0], voxel_size[0], device=bounds.device)
    grid_points = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), axis=-1).float() # (x_size, y_size, z_size, 3)

    # put randomness to the grid points for data augmentation perspective
    if mode == 'train':
        grid_points = grid_points + torch.randn(grid_points.shape, device=bounds.device) * torch.tensor(voxel_size, device=bounds.device)[[2,1,0]]
        # sanitize
        grid_points[:, :, :, 0] = torch.clamp(grid_points[:, :, :, 0], bounds[0,0,0], bounds[0,1,0])
        grid_points[:, :, :, 1] = torch.clamp(grid_points[:, :, :, 1], bounds[0,0,1], bounds[0,1,1])
        grid_points[:, :, :, 2] = torch.clamp(grid_points[:, :, :, 2], bounds[0,0,2], bounds[0,1,2])

    return grid_points[None, :, :, :, :]


# torch version of dataset.load_3d_object_data(self, world_bounds, world_mesh_voxel_out_sh, input_view_R, input_view_t, input_view_K, input_mask, downsample_ratio=1)
def torch_get_object_points_and_voxels(world_bounds: torch.Tensor, input_voxel_size: List[float], input_view_parameters: Dict[str, torch.Tensor], input_view_mask: torch.Tensor, mode: str, downsample_ratio=1):
    """
    world_bounds: (1,2,3) in world
    input_voxel_size: (3,) int # voxel resolution for grid point sampling
    input_view_R: (1, num_input_views, 3, 3)
    input_view_t: (1, num_input_views, 3, 1)
    input_view_K: (1, num_input_views, 3, 3)
    input_view_mask: (1, num_input_views, cfg.input_img_shape[0], cfg.input_img_shape[1])

    // Returns //
    world_inside_object_points: (num_object_points, 3)
    world_object_points_voxel_coord: (num_object_points, 3)
    """
    input_view_R = input_view_parameters['input_view_R'][:, 0]
    input_view_t = input_view_parameters['input_view_t'][:, 0]
    input_view_K = input_view_parameters['input_view_K'][:, 0]
    input_view_mask = input_view_mask[:, 0]

    # get partially true object points from input masks
    world_grid_points = torch_get_grid_points_from_bounds(world_bounds, input_voxel_size, mode)  # (1, x_size, y_size, z_size, 3)
    world_object_inside = torch_prepare_inside_world_object_pts(world_grid_points, input_view_R, input_view_t, input_view_K, input_view_mask)  # np.uint8, [0,1], (x_size, y_size, z_size)
    # handle the exception when there is no inside object points
    if world_object_inside.sum() == 0:
        return None, None

    world_inside_object_points = world_grid_points[world_object_inside.bool()] # (num_inside_object_points, 3), xyz

    # down sample
    world_inside_object_points = world_inside_object_points[None, ::downsample_ratio, :]

    # construct the voxel coordinate in the world coordinate
    dhw = world_inside_object_points[:, :, [2, 1, 0]] # (1,2,3)
    min_dhw = world_bounds[:, 0:1, [2, 1, 0]] # (1,1,3)
    
    world_object_points_voxel_coord = torch.round((dhw - min_dhw) / torch.tensor(input_voxel_size, device=input_view_mask.device)[None, None, :]).int()

    return world_inside_object_points, world_object_points_voxel_coord


  
def Rx(theta):
  return np.array([[1, 0, 0],
                   [ 0, np.cos(theta),-np.sin(theta)],
                   [ 0, np.sin(theta),np.cos(theta)]], dtype=np.float32)
  
def Ry(theta):
  return np.array([[np.cos(theta), 0, np.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-np.sin(theta), 0, np.cos(theta)]], dtype=np.float32)
  
def Rz(theta):
  return np.array([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]], dtype=np.float32)