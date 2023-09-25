import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import image


class RGBDataset(Dataset):
    def __init__(self, dataset_dir, has_gt):
        """
        In:
            dataset_dir: string, train_dir, val_dir, and test_dir in segmentation.py.
                         Be careful the images are stored in the subfolders under these directories.
            has_gt: bool, indicating if the dataset has ground truth masks.
        Out:
            None.
        Purpose:
            Initialize instance variables.
        """
        # TODO: transform to be applied on a sample.
        #  For this homework, compose transforms.ToTensor() and transforms.Normalize() for RGB image should be enough.

        # TODO: number of samples in the dataset.
        #  You'd better not hard code the number,
        #  because this class is used to create train, validation and test dataset (which have different sizes).
        #  Input normalization info to be used in transforms.Normalize()
        mean_rgb = [0.722, 0.751, 0.807]
        std_rgb = [0.171, 0.179, 0.197]

        self.dataset_dir = dataset_dir  # Store the dataset directory.
        self.has_gt = has_gt  # Store whether the dataset has ground truth masks.
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean_rgb, std_rgb)])
        # Create a data transformation pipeline that converts the RGB image to a tensor and applies normalization.

        self.rgb_folder = os.path.join(dataset_dir, 'rgb')  # Store the path to the RGB image folder.
        self.gt_folder = os.path.join(dataset_dir, 'gt')  # Store the path to the ground truth mask folder.
        self.dataset_length = len(os.listdir(self.rgb_folder))  # Store the number of samples in the dataset by counting the number of files in the RGB image folder.


    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        """
        In:
            idx: int, index of each sample, in range(0, dataset_length).
        Out:
            sample: a dictionary that stores paired rgb image and corresponding ground truth mask (if available).
                    rgb_img: Tensor [3, height, width]
                    target: Tensor [height, width], use torch.LongTensor() to convert.
        Purpose:
            Given an index, return paired rgb image and ground truth mask as a sample.
        Hint:
            Use image.read_rgb() and image.read_mask() to read the images.
            Think about how to associate idx with the file name of images.
        """
        # TODO: read RGB image and ground truth mask, apply the transformation, and pair them as a sample.

        # Check if idx is valid
        if idx < 0 or idx >= self.dataset_length:
            raise IndexError(f"Index {idx} is out of range for dataset of length {self.dataset_length}")

        # Read RGB image and apply the transformation
        r_name = f"{idx}_rgb.png"
        rgb_path = os.path.join(self.rgb_folder, r_name)
        if not os.path.isfile(rgb_path):
            raise FileNotFoundError(f"File {rgb_path} not found.")
        rgb_img = image.read_rgb(rgb_path)
        rgb_img = self.transform(rgb_img)

        # If ground truth masks are available, read the mask and convert it to LongTensor
        if self.has_gt:
            g_name = f"{idx}_gt.png"
            gt_path = os.path.join(self.gt_folder, g_name)
            if not os.path.isfile(gt_path):
                raise FileNotFoundError(f"File {gt_path} not found.")
            gt_mask = image.read_mask(gt_path)
            gt_mask = torch.LongTensor(gt_mask)
            sample = {'input': rgb_img, 'target': gt_mask}
        else:
            sample = {'input': rgb_img}

        return sample