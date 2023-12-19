import os
import cv2
import numpy as np
from typing import Optional, Tuple, List

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt




def load_obj(file_name):
    xyz = []
    face_xyz = []
    rgb = []
    obj_file = open(file_name)
    for line in obj_file:
        words = line.split(' ')
        if words[0] == 'v':
            x, y, z = float(words[1]), float(words[2]), float(words[3])
            xyz.append([x, y, z])
            if len(words) > 4:
                r, g, b = float(words[4]), float(words[5]), float(words[6])
                rgb.append([r, g, b])

        elif words[0] == 'f':
            vi_1, vti_1 = words[1].split('/')[:2]
            vi_2, vti_2 = words[2].split('/')[:2]
            vi_3, vti_3 = words[3].split('/')[:2]

            # change 1-based index to 0-based index
            vi_1, vi_2, vi_3 = int(vi_1)-1, int(vi_2)-1, int(vi_3)-1
            face_xyz.append([vi_1, vi_2, vi_3])
        else:
            pass

    return xyz, rgb, face_xyz


def save_obj(v, c=None, f=None, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + f'{v[i][0]:.6f}' + ' ' + f'{v[i][1]:.6f}' + ' ' + f'{v[i][2]:.6f}')

        if c is not None:
            obj_file.write(' ' + str(c[i][0]) + ' ' + str(c[i][1]) + ' ' + str(c[i][2]))
        obj_file.write('\n')

    if f is not None:
        for i in range(len(f)):
            obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()
    

def vis_2d_keypoints(img, kps, alpha=1, kps_vis=None):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=3,
                   color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
        if kps_vis is not None:
            cv2.putText(kp_mask, str(
                kps_vis[i, 0]), p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        else:
            cv2.putText(kp_mask, str(i), p,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_mesh(img, mesh_vertex, alpha=0.5):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=2, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)


# generate the 360 rotating rendering view for the target object, when there is only one camera. If there is no world coordinate, you can pass identity matrix for world2cam
def generate_rotating_rendering_path(world2cam, object_center, num_render_views=60):
    """
    world2cam: (4,4) numpy matrix, transformation matrix that transforms a 3D point in the world coordinate to the camera coordinate
    object_center: 3D location of object center in the camera coordinate 
    num_render_views: number of breakdown for 360 degree
    avg_y_axis: if there are multiple cameras and want to rotate around the average y_axis, use this.
    """

    lower_row = np.array([[0., 0., 0., 1.]])

    object_center_to_camera_origin = -object_center

    # output; list of transformation matrices that transform a 3D point in the world coordinate to a (rotating) camera coordinate
    render_w2c = []
    for theta in np.linspace(0., 2 * np.pi, num_render_views + 1)[:-1]:
        # transformation from original camera to a new camera
        # theta = - np.pi / 6  # 30 degree
        sin, cos = np.sin(theta), np.cos(theta)
        augR = np.eye(3)
        augR[0, 0], augR[2, 2] = cos, cos
        augR[0, 2], augR[2, 0] = sin, -sin

        # rotate around the camera's y-axis, around the object center
        new_camera_origin = augR @ object_center_to_camera_origin + object_center

        # the new camera's z-axis; it should point to the object
        z_axis = object_center - new_camera_origin
        z_axis = z_axis / np.linalg.norm(z_axis)
        # we are trying to rotate around the y-axis' so y-axis remains the same
        y_axis = np.array([0., 1., 0.])

        # get the x-axis of the new camera
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # convert to correct rotation matrix
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # get the transformation of coordiante-axis from the original camera to the new camera == transformation matrix that transforms a 3D point in the new camera coordinate to the original camera coordinate
        # 원래 카메라에서 새로운 카메라로의 좌표축변환행렬
        R_newcam2origcam = np.stack([x_axis, y_axis, z_axis], axis=1)
        newcam2origcam = np.concatenate(
            [R_newcam2origcam, new_camera_origin[:, None]], axis=1)
        newcam2origcam = np.concatenate([newcam2origcam, lower_row], axis=0)

        # transformation matrix that transforms a 3D point in the original camera coordinate to the new camera coordinate
        origcam2newcam = np.linalg.inv(newcam2origcam)

        # transformation matrix that transforms a 3D point in the world camera coordinate to the new camera coordinate
        world2newcam = origcam2newcam @ world2cam

        render_w2c.append(world2newcam.astype(np.float32))

    return render_w2c


# modified from Pytorch3d
# referred to https://logicatcore.github.io/scratchpad/lidar/sensor-fusion/jupyter/2021/04/20/3D-Oriented-Bounding-Box.html
# Returns a rotation R such that `R @ points` has a best fit plane parallel to the xy plane
def get_rotation_to_best_fit_xy(
    points: np.ndarray, centroid: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    points: (N, 3) numpy matrix, points in 3D
    centroid: (1, 3) numpy matrix, their centroid

    // Return //
    rotation: (3, 3) numpy rotation matrix
    """
    if centroid is None:
        centroid = points.mean(axis=0, keepdim=True)

    # normalize translation
    points_centered = points - centroid

    # get covariance matrix of points, and then get the eigen vectors,
    # which are the orthogonal each other, and thus becomes the new basis vectors
    points_covariance = points_centered.transpose(-1, -2) @ points_centered
    # eigen vectors / eigen values are in the ascending order,
    _, evec = np.linalg.eigh(points_covariance)

    # use bigger two eigen vectors, assuming the points are flat
    rotation = np.concatenate(
        [evec[..., 1:], np.cross(evec[..., 1], evec[..., 2])[..., None]], axis=1)

    # `R @ points` should fit plane parallel to the xy plane
    # i.e., rotation transforms the cartesian bassis to the eigen vector basis (좌표축 변환 행렬)
    # i.e., rotation.T transforms the points in the cartesian basis to the eigen vector basis (좌표 변환 행렬)
    rotation = rotation.T

    # in practice, (rotation @ points.T).T
    return rotation


"""
Calculates the signed area / Lévy area of a 2D path. If the path is closed,
i.e. ends where it starts, this is the integral of the winding number over
the whole plane. If not, consider a closed path made by adding a straight
line from the end to the start; the signed area is the integral of the
winding number (also over the plane) with respect to that closed path.

If this number is positive, it indicates in some sense that the path
turns anticlockwise more than clockwise, and vice versa.
"""
# modified from Pytorch3d
# not sure what does this mean


def _signed_area(path: np.ndarray) -> int:
    """
    path: (N, 2) numpy matrix, 2d points.

    // Returns //
    signed_area: scalar
    """
    # This calculation is a sum of areas of triangles of the form
    # (path[0], path[i], path[i+1]), where each triangle is half a
    # parallelogram.
    vector = (path[1:] - path[:1])
    x, y = vector[:, 0], vector[:, 1]
    signed_area = (y[1:] * x[:-1] - x[1:] * y[:-1]).sum() * 0.5
    return signed_area


"""
Simple best fitting of a circle to 2D points. In particular, the circle which
minimizes the sum of the squares of the squared-distances to the circle.

Finds (a,b) and r to minimize the sum of squares (over the x,y pairs) of
    r**2 - [(x-a)**2+(y-b)**2]
i.e.
    (2*a)*x + (2*b)*y + (r**2 - a**2 - b**2)*1 - (x**2 + y**2)

In addition, generates points along the circle. If angles is None (default)
then n_points around the circle equally spaced are given. These begin at the
point closest to the first input point. They continue in the direction which
seems to match the movement of points in points2d, as judged by its
signed area. If `angles` are provided, then n_points is ignored, and points
along the circle at the given angles are returned, with the starting point
and direction as before.

(Note that `generated_points` is affected by the order of the points in
points2d, but the other outputs are not.)
"""
# modified from Pytorch3d
# Returns a fitted 2D circle, which includes center, radius, and generated points


def fit_circle_in_2d(
    points2d, n_points: int = 0, angles: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
   
    points2d: (N, 2) numpy matrix, 2D points
    n_points: number of points to generate on the circle, if angles not given
    angles: optional angles in radians of points to generate.

    // Returns //
    center: (2, ) numpy vector
    radius: scalar
    generated_points: (N', 2) numpy matrix, 2D points
    """

    design = np.concatenate([points2d, np.ones_like(points2d[:, :1])], axis=1)
    rhs = (points2d**2).sum(1)
    n_provided = points2d.shape[0]
    if n_provided < 3:
        raise ValueError(
            f"{n_provided} points are not enough to determine a circle")
    # solve the least sqaure problem
    solution, _, _, _ = np.linalg.lstsq(design, rhs[:, None])
    # solution: (3,1) numpy matrix

    center = solution[:2, 0] / 2  # (2*a, 2*b) / 2 -> (a, b)
    # sqrt(r**2 - a**2 - b**2 + (a**2 + b**2)) = sqrt(r*2) = r
    radius = np.sqrt(solution[2, 0] + (center**2).sum())
    if n_points > 0:
        if angles is not None:
            print("n_points ignored because angles provided")
        else:
            angles = np.linspace(0, 2 * np.pi, n_points).astype(np.float32)

    if angles is not None:
        initial_direction_xy = (points2d[0] - center)
        initial_angle = np.arctan2(
            initial_direction_xy[1], initial_direction_xy[0])

        anticlockwise = _signed_area(points2d) > 0
        if anticlockwise:
            use_angles = initial_angle + angles
        else:
            use_angles = initial_angle - angles

        generated_points = center[None, :] + radius * \
            np.stack([np.cos(use_angles), np.sin(use_angles)], axis=-1)

    else:
        generated_points = points2d

    return center, radius, generated_points


"""
Simple best fit circle to 3D points. Uses circle_2d in the least-squares best fit plane.

In addition, generates points along the circle. If angles is None (default)
then n_points around the circle equally spaced are given. These begin at the
point closest to the first input point. They continue in the direction which
seems to be match the movement of points. If angles is provided, then n_points
is ignored, and points along the circle at the given angles are returned,
with the starting point and direction as before.
"""
# modified from Pytorch3d
# Returns a fitted 3D circle using fitted 2D circle


def fit_circle_in_3d(
    points,
    n_points: int = 0,
    angles: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    points: (N, 3) numpy matrix, 3D points that are assumed to be rotating around a some center
    n_points: number of points to generate on the circle
    angles: optional angles in radians of points to generate.
  
    // Returns //
    circle_points_in_3d: (N', 3) numpy matrix, evenly distribute points on the fitted circle
    normal: (3, ) numpy vector, normal of the circle plane
    """
    centroid = points.mean(0)
    projection_rotation = get_rotation_to_best_fit_xy(points, centroid)
    normal = projection_rotation.T[:, 2]

    projected_points = (projection_rotation @ (points - centroid).T).T

    _, _, circle_points_in_2d = fit_circle_in_2d(
        projected_points[:, :2], n_points=n_points, angles=angles
    )
    if circle_points_in_2d.shape[0] > 0:
        circle_points_in_2d_xy0 = np.concatenate(
            [
                circle_points_in_2d,
                np.zeros_like(circle_points_in_2d[:, :1]),
            ],
            axis=1,
        )
        circle_points_in_3d = (projection_rotation.T @
                               circle_points_in_2d_xy0.T).T + centroid
    else:
        circle_points_in_3d = points

    return circle_points_in_3d, normal

# modified from Pytorch3d
# Returns a normal that is aligned with the up vector of the world coordinate


def _disambiguate_normal(normal, up):
    """
    normal: (3,) numpy vector
    up: (3,) numpy vector
    
    // Returns //
    new_up: (3, ) numpy vector
    """
    flip = np.sign(np.sum(up * normal))
    new_up = normal * flip
    return new_up


def look_at_rotation(camera_centers: np.ndarray, at: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    camera_centers: (N, 3) numpy matrix, locations of cameras in the world coordinate
    at: (1, 3) numpy matrix, a location where cameras should look at
    up: (1, 3) numpy matrix, an up vector that the cameras should have

    // Returns //
    world2cam_R: (N, 3, 3) numpy matrix, transforms a 3D point in the world coordinate to the camera coordinate

    """
    z_axes = at - camera_centers
    z_axes = z_axes / np.linalg.norm(z_axes, axis=1, keepdims=True)
    x_axes = np.cross(up, z_axes, axis=1)
    x_axes = x_axes / np.linalg.norm(x_axes, axis=1, keepdims=True)
    y_axes = np.cross(z_axes, x_axes, axis=1)
    y_axes = y_axes / np.linalg.norm(y_axes, axis=1, keepdims=True)

    is_close = np.isclose(x_axes, np.array(
        0.0), atol=5e-3).all(axis=1, keepdims=True)
    if is_close.any():
        replacement = np.cross(y_axes, z_axes, axis=1)
        replacement = replacement / \
            np.linalg.norm(replacement, axis=1, keepdims=True)
        x_axes = np.where(is_close, replacement, x_axes)

    # (N, 3, 3)
    # convert x and y axes to match the conventional camera
    R = np.concatenate(
        (-x_axes[:, :, None], -y_axes[:, :, None], z_axes[:, :, None]), axis=2)

    world2cam_R = R.transpose(0, 2, 1)

    return world2cam_R


def look_at_view_transform(camera_centers: np.ndarray, at: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    camera_centers: (N, 3) numpy matrix, locations of cameras in the world coordinate
    at: (1, 3) numpy matrix, a location where cameras should look at
    up: (1, 3) numpy matrix, an up vector that the cameras should have

    // Returns //
    R: (N, 3, 3) numpy matrix, world2cam
    t: (N, 3) numpy matrix, world2cam
    """

    R = look_at_rotation(camera_centers, at, up)
    t = - np.matmul(R, camera_centers[:, :, None])[:, :, 0]

    return R, t


# Generate the 360 rotating view, but evenly distributed, from the unevenly rotating camera views
def generate_rendering_path_from_multiviews(camera_centers: np.ndarray, world_object_center: np.ndarray, up: np.ndarray = np.array([0., -1., 0.], dtype=np.float32), num_render_views: int = 60, trajectory_scale: float = 1.1) -> List[np.ndarray]:
    """
    camera_centers: (N, 3) numpy matrix, 3D locations of cameras in the world coordinate
    world_object_center: (3,) numpy vector, 3D location of object center in the world coordinate
    up: (3,) numpy vector, up vector of the world coordinate
    
    // Returns //
    Rts: (num_rendering_views, 4, 4) numpy matrix, transformation matrix that transforms a 3D point in the world coordinate to the rotating camera coordinates
    traj: (num_rendering_views, 3) numpy matrix, new camera locations in the world coordinate
    """

    angles = np.linspace(0, 2.0 * np.pi, num_render_views).astype(np.float32)

    # rendering_camera_centers: (num_rendering_views, 3), normal: (3,)
    rendering_camera_centers, normal = fit_circle_in_3d(
        camera_centers, angles=angles)

    # align the normal to up vector of the world corodinate
    up = _disambiguate_normal(normal, up)

    # scale the distance between the rotating cameras and the object center in the world coordinate
    traj = rendering_camera_centers
    _t_mu = traj.mean(axis=0, keepdims=True)
    traj = (traj - _t_mu) * trajectory_scale + _t_mu

    # point all cameras towards the center of the scene
    Rs, ts = look_at_view_transform(
        traj,
        at=world_object_center[None, :],  # (1, 3)
        up=up[None, :],  # (1, 3)
    )

    # (num_rendering_views, 3, 4)
    Rts = np.concatenate([Rs, ts[:, :, None]], axis=2)
    Rts = np.concatenate([Rts, np.array(
        [[[0., 0., 0., 1.]]], dtype=np.float32).repeat(Rts.shape[0], axis=0)], axis=1)

    return Rts, traj


def make_depth_image(
    depths: np.ndarray,
    masks: np.ndarray,
    max_quantile: float = 0.98,
    min_quantile: float = 0.02,
    min_out_depth: float = 0.1,
    max_out_depth: float = 0.9,
) -> np.ndarray:
    """
    Convert a batch of depth maps to a grayscale image.

    Args:
        depths: A numpy array of shape `(B, H, W, 1)` containing a batch of depth maps.
        masks: A numpy array of shape `(B, H, W, 1)` containing a batch of foreground masks.
        max_quantile: The quantile of the input depth values which will
            be mapped to `max_out_depth`.
        min_quantile: The quantile of the input depth values which will
            be mapped to `min_out_depth`.
        min_out_depth: The minimal value in each depth map will be assigned this color.
        max_out_depth: The maximal value in each depth map will be assigned this color.

    Returns:
        depth_image: A numpy array of shape `(B, H, W, 1)` a batch of grayscale
            depth images.
    """
    normfacs = []
    for d, m in zip(depths, masks):
        ok = (d.reshape(-1) > 1e-6) * (m.reshape(-1) > 0.5)
        if ok.sum() <= 1:
            print("Empty depth!")
            normfacs.append(np.zeros(2))
            continue
        dok = d.reshape(-1)[ok].reshape(-1)
        _maxk = max(int(round((1 - max_quantile) * (dok.size))), 1)
        _mink = max(int(round(min_quantile * (dok.size))), 1)
        sorted_dok = np.sort(dok)
        normfac_max = sorted_dok[-_maxk]
        normfac_min = sorted_dok[_mink - 1]
        normfacs.append(np.stack([normfac_min, normfac_max]))
    normfacs = np.stack(normfacs)
    _min, _max = (normfacs[:, 0].reshape(-1, 1, 1, 1),
                  normfacs[:, 1].reshape(-1, 1, 1, 1))
    depths = (depths - _min) / (_max - _min).clip(1e-4)
    depths = (
        (depths * (max_out_depth - min_out_depth) + min_out_depth) * masks.astype(np.float32)
    ).clip(0.0, 1.0)
    return depths
