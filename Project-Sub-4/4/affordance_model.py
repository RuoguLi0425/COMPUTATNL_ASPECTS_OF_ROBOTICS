from typing import Tuple, Optional, Dict

import numpy as np
from matplotlib import cm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

from common import draw_grasp
from collections import deque



def get_gaussian_scoremap(
        shape: Tuple[int, int],
        keypoint: np.ndarray,
        sigma: float=1, dtype=np.float32) -> np.ndarray:
    """
    Generate a image of shape=:shape:, generate a Gaussian distribtuion
    centered at :keypont: with standard deviation :sigma: pixels.
    keypoint: shape=(2,)
    """
    coord_img = np.moveaxis(np.indices(shape),0,-1).astype(dtype)
    sqrt_dist_img = np.square(np.linalg.norm(
        coord_img - keypoint[::-1].astype(dtype), axis=-1))
    scoremap = np.exp(-0.5/np.square(sigma)*sqrt_dist_img)
    return scoremap

class AffordanceDataset(Dataset):
    """
    Transformational dataset.
    raw_dataset is of type train.RGBDataset
    """
    def __init__(self, raw_dataset: Dataset):
        super().__init__()
        self.raw_dataset = raw_dataset

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Transform the raw RGB dataset element into
        training targets for AffordanceModel.
        return:
        {
            'input': torch.Tensor (3,H,W), torch.float32, range [0,1]
            'target': torch.Tensor (1,H,W), torch.float32, range [0,1]
        }
        Note: self.raw_dataset[idx]['rgb'] is torch.Tensor (H,W,3) torch.uint8
        """
        # checkout train.RGBDataset for the content of data
        data = self.raw_dataset[idx]
        # TODO: (problem 2) complete this method and return the correct input and target as data dict
        # Hint: Use get_gaussian_scoremap
        # Hint: https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html
        # ===============================================================================
        ImageData = np.array(data['rgb'])

        RGBshape =ImageData.shape
        Height_image = RGBshape[0]
        Width_image = RGBshape[1]

        Center_Point_X = data['center_point'][0]
        Center_Point_Y = data['center_point'][1]
        Angle = data['angle']


        KPS = KeypointsOnImage([Keypoint(x = Center_Point_X, y = Center_Point_Y),], shape=RGBshape)

        Scalar_Angle = Angle.item() ##Tensor angle transfer to scalar angle
        Angle_increment = 22.5
        binned_angle = np.argmin(np.abs(np.arange(0, 180, Angle_increment) - Scalar_Angle)) * Angle_increment

        seq = iaa.Sequential([iaa.Rotate(-binned_angle.item())])

        image_aug, kps_aug = seq(image=ImageData, keypoints=KPS)

        X_aug = kps_aug[0].x
        Y_aug = kps_aug[0].y
        KeypointDate = np.array([X_aug,Y_aug])

        target = get_gaussian_scoremap((Height_image,Width_image),KeypointDate)
        input = torch.from_numpy(image_aug)

        TargetData = np.expand_dims(target, axis=0)
        InputData = input.permute(2,0,1).type(torch.float32)## swap the original sequence:(H,W,C) TO (C,H,W). Because it is the Pytorch expected output.

        data = {'input': InputData, 'target': TargetData}
        return data

class AffordanceModel(nn.Module):
    def __init__(self, n_channels: int=3, n_classes: int=1, n_past_actions: int=0, **kwargs):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = nn.Sequential(
            # For simplicity:
            #     use padding 1 for 3*3 conv to keep the same Width and Height and ease concatenation
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.outc = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
        # hack to get model device
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.past_actions = deque(maxlen=n_past_actions)

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_inc = self.inc(x)  # N * 64 * H * W
        x_down1 = self.down1(x_inc)  # N * 128 * H/2 * W/2
        x_down2 = self.down2(x_down1)  # N * 256 * H/4 * W/4
        x_up1 = self.upconv1(x_down2)  # N * 128 * H/2 * W/2
        x_up1 = torch.cat([x_up1, x_down1], dim=1)  # N * 256 * H/2 * W/2
        x_up1 = self.conv1(x_up1)  # N * 128 * H/2 * W/2
        x_up2 = self.upconv2(x_up1)  # N * 64 * H * W
        x_up2 = torch.cat([x_up2, x_inc], dim=1)  # N * 128 * H * W
        x_up2 = self.conv2(x_up2)  # N * 64 * H * W
        x_outc = self.outc(x_up2)  # N * n_classes * H * W
        return x_outc

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict affordance using this model.
        This is required due to BCEWithLogitsLoss
        """
        return torch.sigmoid(self.forward(x))

    @staticmethod
    def get_criterion() -> torch.nn.Module:
        """
        Return the Loss object needed for training.
        Hint: why does nn.BCELoss does not work well?
        """
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray,
            target: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Visualize rgb input and affordance as a single rgb image.
        """
        cmap = cm.get_cmap('viridis')
        in_img = np.moveaxis(input, 0, -1)
        pred_img = cmap(output[0])[...,:3]
        row = [in_img, pred_img]
        if target is not None:
            gt_img = cmap(target[0])[...,:3]
            row.append(gt_img)
        img = (np.concatenate(row, axis=1)*255).astype(np.uint8)
        return img

    def predict_grasp(
            self,
            rgb_obs: np.ndarray,
    ) -> Tuple[Tuple[int, int], float, np.ndarray]:
        """
        Given an RGB image observation, predict the grasping location and angle in image space.
        return coord, angle, vis_img
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.

        Note: torchvision's rotation is counter clockwise, while imgaug,OpenCV's rotation are clockwise.
        """
        device = self.device
        # TODO: (problem 2) complete this method (prediction)
        # Hint: why do we provide the model's device here?
        # ===============================================================================
        num_rotations = 8
        top_n = 1
        rotated_images = []

        if not isinstance(rgb_obs, np.ndarray):
            raise ValueError("Input image must be a numpy array")

        for idx in range(num_rotations):
            transformation_sequence = iaa.Sequential([iaa.Rotate(idx * -22.5)])
            rotated_rgb = transformation_sequence(image=rgb_obs)
            rotated_images.append(rotated_rgb)

        tensor_input = torch.from_numpy(np.stack(rotated_images)).permute(0, 3, 1, 2).type(torch.float32).to(device)
        with torch.no_grad():
            grasp_predictions = self.predict(tensor_input)
        best_prediction = torch.topk(grasp_predictions.flatten(), top_n)[0][-1].item()
        indices = ((grasp_predictions == best_prediction).nonzero())[0]
        # ===============================================================================
        # TODO: (problem 3, skip when finishing problem 2) avoid selecting the same failed actions
        # ===============================================================================
        rot_bin = indices[0].item()
        coord = (indices[3].item(), indices[2].item())
        angle = rot_bin * -22.5
        self.past_actions.append((rot_bin, coord, angle)) ## create a list to append past actions. if action is fail, next action will avoid chosing from fail action list

        # Suppress the past action's location
        for max_coord in list(self.past_actions):
            bin = max_coord[0]
            NegtiveMap = get_gaussian_scoremap(shape=grasp_predictions[bin].shape[-2:],keypoint=np.array(max_coord[1]), sigma=1)
            grasp_predictions[bin] -= torch.from_numpy(NegtiveMap).to(device) ##Subtract the negative Gaussian score map from the predicted output to suppress the past action's location

        # ===============================================================================
        # TODO: (problem 2) complete this method (visualization)
        # :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.
        # Hint: use common.draw_grasp
        # Hint: see self.visualize
        # Hint: draw a grey (127,127,127) line on the bottom row of each image.
        # ===============================================================================
        grasp_coordinates = (indices[3].item(), indices[2].item())
        averge_angle = 180/8
        # calculate the average angle between each rotation in degrees

        inverse_averge_angle = -averge_angle
        rotation_angle = indices[0].item() * inverse_averge_angle

        tensor_input_np = np.array(tensor_input.cpu())
        P_out_np = np.array(grasp_predictions.cpu())
        vis_imgs_list = list()
        selected_input = np.empty(0)

        idx = 0
        while idx < num_rotations:
            input_img = tensor_input_np[idx,:,:,:]
            target_img = P_out_np[idx,:,:,:]
            vis_img = self.visualize(input_img, target_img)
            vis_img[127, :, :] = 127
            vis_imgs_list.append(vis_img)

            if idx * -22.5 == rotation_angle:
                draw_grasp(vis_img, grasp_coordinates, 0.0) # draw a grasp on the visualized image at the selected angle
                selected_input = np.moveaxis(input_img, 0, -1) # select the input image at the selected angle

            idx += 1 ## pass all direction(8 directions because angle split from 0 to 315)

            if idx * -22.5 == rotation_angle:
                draw_grasp(vis_img, grasp_coordinates, 0.0)
                selected_input = np.moveaxis(input_img, 0, -1)

        vis_img = np.vstack([
            np.hstack([vis_imgs_list[0], vis_imgs_list[1]]),
            np.hstack([vis_imgs_list[2], vis_imgs_list[3]]),
            np.hstack([vis_imgs_list[4], vis_imgs_list[5]]),
            np.hstack([vis_imgs_list[6], vis_imgs_list[7]]),
                            ]) # concatenate the visualized images into a single image

        keypoints = KeypointsOnImage([Keypoint(x=grasp_coordinates[0], y=grasp_coordinates[1]), ],shape=selected_input.shape)
        # create keypoints for the selected input

        transformation_sequence = iaa.Sequential([iaa.Rotate(-rotation_angle)])
        # create a rotation transformation sequence for the selected input

        augmented_image, augmented_keypoints = transformation_sequence(image=selected_input, keypoints=keypoints)
        # apply the rotation transformation to the selected input and keypoints

        augmented_x = int(augmented_keypoints[0].x) # get the x-coordinate of the rotated grasp
        augmented_y = int(augmented_keypoints[0].y) # get the y-coordinate of the rotated grasp
        coord = (augmented_x, augmented_y)
        # print(augmented_x,augmented_y)


        if not isinstance(coord, tuple):
            raise ValueError("Final coordinates must be a tuple")


        angle = -rotation_angle
        # ===============================================================================
        return coord, angle, vis_img


