import argparse
import os

import cv2
import numpy as np
import trimesh

import image
from transforms import depth_to_point_cloud, transform_point3s
from camera import Camera, cam_view2pose

parser = argparse.ArgumentParser()
parser.add_argument('--val', action='store_true', help='pose estimation for validation set')
parser.add_argument('--test', action='store_true', help='pose estimation for test set')

LIST_OBJ_FOLDERNAME = [
        "004_sugar_box",  # obj_id == 1
        "005_tomato_soup_can",  # obj_id == 2
        "007_tuna_fish_can",  # obj_id == 3
        "011_banana",  # obj_id == 4
        "024_bowl",  # obj_id == 5
    ]


def obj_mesh2pts(obj_id, point_num, transform=None):
    """
    In:
        obj_id: int, indicating an object in LIST_OBJ_FOLDERNAME.
        point_num: int, number of points to sample.
        transform: Numpy array [4, 4] of float64.
    Out:
        pts: Numpy array [n, 3], sampled point cloud.
    Purpose:
         Sample a point cloud from the mesh of the specific object. If transform is not None, apply it.
    """
    mesh_path = './YCB_subsubset/' + LIST_OBJ_FOLDERNAME[obj_id - 1] + '/model_com.obj'  # objects ID start from 1
    mesh = trimesh.load(mesh_path)
    if transform is not None:
        mesh = mesh.apply_transform(transform)
    pts, _ = trimesh.sample.sample_surface(mesh, count=point_num)
    return pts


def gen_obj_depth(obj_id, depth, mask):
    """
    In:
        obj_id: int, indicating an object in LIST_OBJ_FOLDERNAME.
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
    Out:
        obj_depth: Numpy array [height, width] of float64, where depth value of all the pixels that don't belong to the object is 0.
    Purpose:
        Generate depth image for a specific object given obj_id.
        Generate depth for all objects when obj_id == -1. You should filter out the depth of the background, where the ID is 0 in the mask. We want to preserve depth only for object 1 to 5 inclusive.
    """
    # TODO
    obj_depth = np.copy(depth)

    if obj_id != -1:
        obj_depth[mask != obj_id] = 0
    else:
        obj_depth[(mask < 1) | (mask > 5)] = 0

    return obj_depth


def obj_depth2pts(obj_id, depth, mask, camera, view_matrix):
    """
    In:
        obj_id: int, indicating an object in LIST_OBJ_FOLDERNAME.
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
        camera: Camera instance.
        view_matrix: Numpy array [16,] of float64, representing a 4x4 matrix.
    Out:
        world_pts: Numpy array [n, 3], 3D points in the world frame of reference.
    Purpose:
        Generate point cloud projected from depth of the specific object(s) in the world frame of reference.
    Hint:
        The imported depth_to_point_cloud(), cam_view2pose() and transform_point3s() can be useful here.
        The view matrices are provided in the /dataset/val/view_matrix or /dataset/test/view_matrix folder.
    """
    # TODO

    obj_depth = gen_obj_depth(obj_id, depth, mask)    # Generate the depth image for the specific object(s)

    camera_pts = depth_to_point_cloud(camera.intrinsic_matrix, obj_depth)    # Convert the depth image to a point cloud in the camera's frame of reference

    wTc = cam_view2pose(view_matrix)    # Convert the camera view matrix to a camera pose (world-to-camera transformation)

    world_pts = transform_point3s(wTc, camera_pts)    # Transform the point cloud from the camera's frame of reference to the world frame of reference

    return world_pts


def align_pts(pts_a, pts_b, max_iterations=20, threshold=1e-05):
    """
    In:
        pts_a: Numpy array [n, 3].
        pts_b: Numpy array [n, 3].
        max_iterations: int, tunable parameter of trimesh.registration.icp().
        threshold: float, tunable parameter of trimesh.registration.icp().
    Out:
        matrix: Numpy array [4, 4], the transformation matrix sending pts_a to pts_b.
    Purpose:
        Apply the iterative closest point algorithm to estimate a transformation that aligns one point cloud with another.
    Hint:
        Use trimesh.registration.icp() and trimesh.registration.procrustes().
        scale=False and reflection=False should be passed to both icp() and procrustes().
    """
    # TODO
    # Check if pts_a is empty
    if len(pts_a) == 0:
        return None

    try:
        # Calculate Procrustes transformation
        matrix, transformed, cost = trimesh.registration.procrustes(pts_a, pts_b, reflection=False, scale=False)
    except np.linalg.LinAlgError:
        # If there's a linear algebra error, return None
        return None

    # Perform ICP (iterative closest point) algorithm
    icp_matrix, icp_transformed, icp_cost = trimesh.registration.icp(
        pts_a, pts_b, initial=matrix, threshold=threshold,
        max_iterations=max_iterations, reflection=False, scale=False
    )

    # Check if the cost of ICP is above a certain threshold value
    if icp_cost > 1e-03:
        print("Warning: High ICP cost, the alignment might be inaccurate.")

    return icp_matrix


def estimate_pose(depth, mask, camera, view_matrix):
    """
    In:
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
        camera: Camera instance.
        view_matrix: Numpy array [16,] of float64, representing a 4x4 matrix.
    Out:
        list_obj_pose: a list of transformation matrices (Numpy array [4, 4] of float64).
                       The order is the same as in LIST_OBJ_FOLDERNAME,
                       so list_obj_pose[i] is the pose of the object with obj_id == i+1.
                       If the pose of an object is missing, the item should be None.
    Purpose:
        Perform pose estimation on each object in the given image.
    """
    # TODO
    list_obj_pose = list()

    # to change
    max_iterations= 40
    threshold= 1e-07

    i = 0
    while i < 5:
        obj_id = i + 1
        # Get projected points and sample points
        proj_pnts = obj_depth2pts(obj_id, depth, mask, camera, view_matrix)
        n = proj_pnts.shape[0]
        sample_pnts = obj_mesh2pts(obj_id, n)

        # Align the two sets of points
        matrix = align_pts(sample_pnts, proj_pnts, max_iterations=max_iterations, threshold=threshold)

        # Append transformation matrix to the list, or None if failed
        if matrix is None:
            list_obj_pose.append(None)
            print("Failed to estimate pose for object ", obj_id)
        else:
            list_obj_pose.append(matrix)

        # Move to the next object
        i = i + 1

    return list_obj_pose


def save_pose(dataset_dir, folder, scene_id, list_obj_pose):
    """
    In:
        dataset_dir: string, path of the val or test folder.
        folder: string, the folder to save the pose.
                "gtmask" -- for pose estimated using ground truth mask
                "predmask" -- for pose estimated using predicted mask
        scene_id: int, ID of the scene.
        list_obj_pose: a list of transformation matrices (Numpy array [4, 4] of float64).
                       The order is the same as in LIST_OBJ_FOLDERNAME,
                       so list_obj_pose[i] is the pose of the object with obj_id == i+1.
                       If the pose of an object is missing, the item would be None.
    Out:
        None.
    Purpose:
        Save the pose of each object in a scene.
    """
    pose_dir = dataset_dir + "pred_pose/" + folder + "/"
    print(f"Save poses as .npy files to {pose_dir}")
    for i in range(len(list_obj_pose)):
        pose = list_obj_pose[i]
        if pose is not None:
            np.save(pose_dir + str(scene_id) + "_" + str(i + 1), pose)


def export_gt_ply(scene_id, depth, gt_mask, camera, view_matrix):
    """
    In:
        scene_id: int, ID of the scene.
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
        camera: Camera instance.
        view_matrix: Numpy array [16,] of float64, representing a 4x4 matrix.
    Out:
        None.
    Purpose:
        Export a point cloud of the ground truth scene -- projected from depth using ground truth mask-- with the color green.
    """
    print("Export gt point cloud as .ply file to ./dataset/val/exported_ply/")
    file_path = "./dataset/val/exported_ply/" + str(scene_id) + "_gtmask.ply"
    pts = obj_depth2pts(-1, depth, gt_mask, camera, view_matrix)
    if len(pts) == 0:
        print("Empty point cloud!")
    else:
        ptcloud = trimesh.points.PointCloud(vertices=pts, colors=[0, 255, 0])  # Green
        ptcloud.export(file_path)


def export_pred_ply(dataset_dir, scene_id, suffix, list_obj_pose):
    """
    In:
        dataset_dir: string, path of the val or test folder.
        scene_id: int, ID of the scene.
        suffix: string, indicating which kind of point cloud is going to be exported.
                "gtmask_transformed" -- transformed with pose estimated using ground truth mask
                "predmask_transformed" -- transformed with pose estimated using prediction mask
        list_obj_pose: a list of transformation matrices (Numpy array [4, 4] of float64).
                       The order is the same as in LIST_OBJ_FOLDERNAME,
                       so list_obj_pose[i] is the pose of the object with obj_id == i+1.
                       If the pose of an object is missing, the item would be None.
    Out:
        None.
    Purpose:
        Export a point cloud of the predicted scene with single color.
    """
    ply_dir = dataset_dir + "exported_ply/"
    print(f"Export predicted point cloud as .ply files to {ply_dir}")
    file_path = ply_dir + str(scene_id) + "_" + suffix + ".ply"
    color_switcher = {
        "gtmask_transformed": [0, 0, 255],  # Blue
        "predmask_transformed": [255, 0, 0],  # Red
    }
    pts = np.empty([0, 3])  # Numpy array [n, 3], the point cloud to be exported.
    for obj_id in range(1, 6):  # obj_id indicates an object in LIST_OBJ_FOLDERNAME
        pose = list_obj_pose[obj_id - 1]
        if pose is not None:
            obj_pts = obj_mesh2pts(obj_id, point_num=1000, transform=pose)
            pts = np.concatenate((pts, obj_pts), axis=0)
    if len(pts) == 0:
        print("Empty point cloud!")
    else:
        ptcloud = trimesh.points.PointCloud(vertices=pts, colors=color_switcher[suffix])
        ptcloud.export(file_path)


def main():
    args = parser.parse_args()
    if args.val:
        dataset_dir = "./dataset/val/"
        print("Pose estimation for validation set")
    elif args.test:
        dataset_dir = "./dataset/test/"
        print("Pose estimation for test set")
    else:
        print("Missing argument --val or --test")
        return

    # Setup camera -- to recover coordinate, keep consistency with that in gen_dataset.py
    my_camera = Camera(
        image_size=(240, 320),
        near=0.01,
        far=10.0,
        fov_width=69.40
    )

    if not os.path.exists(dataset_dir + "exported_ply/"):
        os.makedirs(dataset_dir + "exported_ply/")
    if not os.path.exists(dataset_dir + "pred_pose/"):
        os.makedirs(dataset_dir + "pred_pose/")
        os.makedirs(dataset_dir + "pred_pose/predmask/")
        if args.val:
            os.makedirs(dataset_dir + "pred_pose/gtmask/")

    # TODO:
    #  Use the implemented estimate_pose() to estimate the pose of the objects in each scene of the validation set and test set.
    #  For the validation set, use both ground truth mask and predicted mask.
    #  For the test set, use the predicted mask.
    #  Use save_pose(), export_gt_ply() and export_pred_ply() to generate files to be submitted.
    for scene_id in range(5):
        print("Estimating scene", scene_id)
        # TODO
        if args.val:
            # Load depth, mask, and view matrix from file
            depth_path = dataset_dir + 'depth/' + str(scene_id) + '_depth.png'
            mask_path = dataset_dir + 'gt/' + str(scene_id) + '_gt.png'
            view_matrix_path = dataset_dir + 'view_matrix/' + str(scene_id) + '.npy'

            if not os.path.exists(depth_path):
                print("Missing depth file:", depth_path)
                continue
            if not os.path.exists(mask_path):
                print("Missing mask file:", mask_path)
                continue
            if not os.path.exists(view_matrix_path):
                print("Missing view matrix file:", view_matrix_path)
                continue

            # Read depth, mask, and view matrix
            depth = image.read_depth(depth_path)
            mask = image.read_mask(mask_path)
            view_matrix = np.load(view_matrix_path)

            # Export the ground truth point cloud
            export_gt_ply(scene_id, depth, mask, my_camera, view_matrix)

            # Estimate pose with ground truth mask
            list_obj_pose = estimate_pose(depth, mask, my_camera, view_matrix)
            if list_obj_pose is None:
                print("Could not estimate pose for scene", scene_id)
                continue

            # Save pose and transformed point cloud
            export_pred_ply(dataset_dir, scene_id, 'gtmask_transformed', list_obj_pose)
            save_pose(dataset_dir, 'gtmask', scene_id, list_obj_pose)

            # Estimate pose with predicted mask
            mask_path = dataset_dir + 'pred/' + str(scene_id) + '_pred.png'
            if not os.path.exists(mask_path):
                print("Missing predicted mask file:", mask_path)
                continue
            mask = image.read_mask(mask_path)
            list_obj_pose = estimate_pose(depth, mask, my_camera, view_matrix)
            if list_obj_pose is None:
                print("Could not estimate pose for scene", scene_id)
                continue

            # Save pose and transformed point cloud
            export_pred_ply(dataset_dir, scene_id, 'predmask_transformed', list_obj_pose)
            save_pose(dataset_dir, 'predmask', scene_id, list_obj_pose)


        elif args.test:

            depth_path = dataset_dir+'depth/'+str(scene_id)+'_depth.png'

            mask_path = dataset_dir+'pred/'+str(scene_id)+'_pred.png'

            view_matrix_path = dataset_dir+'view_matrix/'+str(scene_id)+'.npy'
            if not os.path.exists(depth_path) or not os.path.isfile(depth_path):
                print("Error: could not read depth image for scene", scene_id)
                continue

            if not os.path.exists(mask_path) or not os.path.isfile(mask_path):
                print("Error: could not read predicted mask image for scene", scene_id)
                continue

            if not os.path.exists(view_matrix_path) or not os.path.isfile(view_matrix_path):
                print("Error: could not read view matrix for scene", scene_id)
                continue

            depth = image.read_depth(depth_path)
            mask = image.read_mask(mask_path)
            view_matrix= np.load(view_matrix_path)

            # Estimate the pose of each object in the scene using the depth image and predicted mask
            list_obj_pose= estimate_pose(depth, mask, my_camera, view_matrix)

            # Export the transformed PLY file and save the pose to a text file
            export_pred_ply(dataset_dir, scene_id, 'predmask_transformed', list_obj_pose)
            save_pose(dataset_dir, 'predmask', scene_id, list_obj_pose)


if __name__ == '__main__':
    main()