from argparse import Namespace
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import ImageOps, Image
from dreifus.image import Img
from elias.util import load_img
from elias.util.io import resize_img
from torch import nn

from gan3d_inversion.triplane_net.models.triplanenet_v2 import TriPlaneNet
from torchvision.transforms.functional import normalize


class TriplaneNetModel(nn.Module):

    def __init__(self, device: torch.device = torch.device('cuda')):
        super().__init__()
        checkpoint_path = f"{Path.home()}/.cache/TriplaneNet/triplanenet_v2_final.pth"
        ckpt = torch.load(checkpoint_path, map_location=device)
        opts = ckpt['opts']
        opts['checkpoint_path'] = checkpoint_path
        opts = Namespace(**opts)
        self._net = TriPlaneNet(opts).to(device)
        self._device = device

    def to(self, device: torch.device):
        self._net.to(device)
        self._device = device

    @torch.no_grad()
    def invert(self, image: np.ndarray, camera: np.ndarray):
        img = resize_img(image, 0.5)

        pose, intrinsics = np.array(camera[:, :16]).reshape(4, 4), np.array(camera[:, 16:]).reshape(3, 3)
        flipped_pose = flip_yaw(pose)
        mirror_camera = np.concatenate([flipped_pose.reshape(-1), intrinsics.reshape(-1)])[None]

        mirror_image = np.array(ImageOps.mirror(Image.fromarray(img)))
        img = Img.from_numpy(img).to_torch().img.to(self._device)
        mirror_image = Img.from_numpy(mirror_image).to_torch().img.to(self._device)
        mirror_image = normalize(mirror_image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        img = normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        camera = torch.tensor(camera).float().to(self._device)
        mirror_camera = torch.tensor(mirror_camera).float().to(self._device)

        output = self._net.forward(img[None], camera, mirror_camera, x_mirror=mirror_image[None])

        return output

    @torch.no_grad()
    def render(self, output: dict, cameras: torch.Tensor) -> List[np.ndarray]:
        triplane_offsets = output['triplane_offsets']
        latent_codes = output['latent_codes']
        B = len(cameras)
        triplane_offsets = triplane_offsets.repeat(B, 1, 1, 1)
        latent_codes = latent_codes.repeat(B, 1, 1)
        render_outputs = self._net.decoder.synthesis(latent_codes, cameras.clone().detach(), triplane_offsets=triplane_offsets, noise_mode='const')
        images = [Img.from_normalized_torch(img).to_numpy().img for img in render_outputs['image']]

        return images


def flip_yaw(pose_matrix):
    flipped = pose_matrix.copy()
    flipped[0, 1] *= -1
    flipped[0, 2] *= -1
    flipped[1, 0] *= -1
    flipped[2, 0] *= -1
    flipped[0, 3] *= -1
    return flipped
