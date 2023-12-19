import open3d as o3d
import numpy as np


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

if __name__ == '__main__':
    panda_hand_path = "./resources_panda_meshes_collision_hand.stl"
    panda_finger_path = "./resources_panda_meshes_collision_finger.stl"
    
    hand_mesh = o3d.io.read_triangle_mesh(panda_hand_path)
    finger_mesh = o3d.io.read_triangle_mesh(panda_hand_path)
    hand_mesh.compute_vertex_normals()
    finger_mesh.compute_vertex_normals()

    # transform to the gripper
    gripper_vertices = np.concatenate([np.asarray(hand_mesh.vertices), np.asarray(finger_mesh.vertices)])
    gripper_contact = gripper_vertices.min(0)[None, :]# gripper_vertices.mean(0, keepdims=True)

    gt= True
    object_name = 'Drill'
    method_name = 'HandNeRF'
    if gt:
        obj_path = f'/home/hongsuk.c/Downloads/MeshForPathPlanning/DexYCB_{object_name}/HandNeRF/gt_object_mesh.obj'
        vertices, rgb, faces = load_obj(obj_path)


        vertices = np.array(vertices)

        vertices = vertices - vertices.mean(0, keepdims=True) + gripper_contact

        object_mesh = o3d.geometry.TriangleMesh()
        object_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        object_mesh.triangles = o3d.utility.Vector3iVector(faces)

        save_path = f"hand_with_GT_{object_name}.stl"

    else:
        obj_path = f'/home/hongsuk.c/Downloads/MeshForPathPlanning/DexYCB_{object_name}/{method_name}/voxelsize_0.002_thr10_object.ply'
        object_mesh = o3d.io.read_triangle_mesh(obj_path)

        vertices = object_mesh.vertices
        vertices = np.array(vertices)

        vertices = vertices - vertices.mean(0, keepdims=True) + gripper_contact
        object_mesh.vertices = o3d.utility.Vector3dVector(vertices)

        save_path = f"hand_with_{method_name}_{object_name}.stl"

    # transform to the gripper

    gripper_real_center = gripper_vertices.mean(0)
    object_mesh.rotate(object_mesh.get_rotation_matrix_from_xyz((0,-np.pi/2,np.pi/2)), center=gripper_real_center)

    object_mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([hand_mesh, finger_mesh, object_mesh])

    # save new hand mesh
    new_hand_mesh = object_mesh + hand_mesh
    o3d.io.write_triangle_mesh(save_path, new_hand_mesh)


    pass