import cv2
import numpy as np


def write_grayscale(image, file_path):
    """Write out a grayscale image.

    Args:
        image (numpy.array [h, w]): array representing the grayscale image
        file_path (str): out path to write image, including extention
    """
    cv2.imwrite(file_path, image)


def read_grayscale(file_path):
    """Read in a grayscale image.

    Args:
        file_path (str): image path to read in.

    Returns:
        numpy.array [h, w]:  Grayscale image as array, each value in range [0, 255].
    """
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)


def write_rgb(image, file_path):
    """Write the RGB image.

    Args:
        image (numpy.array [h, w, 3]): array representing the rgb image
        file_path (str): out path to write image, including extention
    """
    cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def read_rgb(file_path):
    """Read in a color image.

    Args:
        file_path (str): Color image to read.

    Returns:
        np.array [h, w, 3]:  Grayscale image as array, each entry is an r, g, or b value in range [0, 255].
            Note: channel order is r, then g, then b.
    """
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)


def write_depth(depth_image, file_path):
    """Write a depth image to a 16-bit png. Store in mm to preserve precision.

    Args:
        depth_image (numpy.array [h, w]): Each value is z depth in meters.
        file_path (str): Output png file path.
    """
    # convert from depth in meters to millimeters
    depth_image = depth_image * 1000.

    depth_image = depth_image.astype(np.uint16)
    cv2.imwrite(file_path, depth_image)


def read_depth(file_path):
    """Read in a 16-bit png depth image (mm scale).

    Args:
        file_path (str): Path to image.

    Returns:
        np.array [h, w]: Array where each value is a z depth in meters.
    """
    # depth is saved as 16-bit uint in millimenters
    depth_image = cv2.imread(file_path, -1).astype(float)

    # millimeters to meters
    depth_image /= 1000.

    return depth_image
