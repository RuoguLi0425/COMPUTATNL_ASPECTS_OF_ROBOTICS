from image import read_rgb, read_depth
import numpy as np
import os
from ply import Ply
import time
import tsdf

if __name__ == "__main__":
    # Set bounds based on max and min in each dimension in the world space.
    image_count = 10
    camera_intrensics = np.loadtxt("./data/camera-intrinsics.txt", delimiter=' ')
    volume_bounds = np.array([[-0.75,  0.75], [-0.75, 0.75], [0., 0.8]])

    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_volume = tsdf.TSDFVolume(volume_bounds, voxel_size=0.01)

    # Loop through RGB-D images and fuse them together
    start_time = time.time()
    for i in range(image_count):
        print("Fusing frame %d/%d"%(i+1, image_count))

        # Read RGB-D image and camera pose
        color_image = read_rgb("./data/frame-%06d.color.png"%(i))
        depth_image = read_depth("./data/frame-%06d.depth.png"%(i))
        camera_pose = np.loadtxt("./data/frame-%06d.pose.txt"%(i))

        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_volume.integrate(color_image, depth_image, camera_intrensics, camera_pose, observation_weight=1.)

    fps = image_count / (time.time() - start_time)
    print("Average FPS: {:.2f}".format(fps))

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...")
    points, faces, normals, colors = tsdf_volume.get_mesh()
    mesh = Ply(triangles=faces, points=points, normals=normals, colors=colors)
    mesh.write(os.path.join('supplemental', 'mesh.ply'))

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud to point_cloud.ply...")
    pc = Ply(points=points, normals=normals, colors=colors)
    pc.write(os.path.join('supplemental', 'point_cloud.ply'))
