from argparse import Namespace
from pathlib import Path
from unittest import TestCase

import numpy as np
import torch
from PIL import ImageOps, Image
from dreifus.image import Img
from elias.util import load_img
from elias.util.io import resize_img
from torchvision.transforms.functional import normalize

from gan3d_inversion.goae.camera_utils import create_camera_for_yaw
from gan3d_inversion.triplane_net.models.triplanenet_model import TriplaneNetModel
from gan3d_inversion.triplane_net.models.triplanenet_v2 import TriPlaneNet

def flip_yaw(pose_matrix):
    flipped = pose_matrix.copy()
    flipped[0, 1] *= -1
    flipped[0, 2] *= -1
    flipped[1, 0] *= -1
    flipped[2, 0] *= -1
    flipped[0, 3] *= -1
    return flipped
class TriplaneNetTest(TestCase):
    def test_triplane_net(self):
        checkpoint_path = f"{Path.home()}/.cache/TriplaneNet/triplanenet_v2_final.pth"
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = checkpoint_path
        # opts.update(vars(test_opts))
        opts = Namespace(**opts)
        net = TriPlaneNet(opts)

        name = "meta_fairy_tale_woman"
        real_label = np.load(f"D:/Projects/eg3d-preprocessor/test/dataset/c/{name}.npy")[None]
        real_img_512 = load_img(f"D:/Projects/eg3d-preprocessor/test/dataset/crop/{name}.jpg")
        img = resize_img(real_img_512, 0.5)

        label = real_label
        pose, intrinsics = np.array(label[:, :16]).reshape(4, 4), np.array(label[:, 16:]).reshape(3, 3)
        flipped_pose = flip_yaw(pose)
        mirror_label = np.concatenate([flipped_pose.reshape(-1), intrinsics.reshape(-1)])[None]

        mirror_image = np.array(ImageOps.mirror(Image.fromarray(img)))
        img = Img.from_numpy(img).to_torch().img.cuda()
        mirror_image = Img.from_numpy(mirror_image).to_torch().img.cuda()
        mirror_image = normalize(mirror_image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        img = normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        camera = torch.tensor(real_label).float().cuda()
        mirror_camera = torch.tensor(mirror_label).float().cuda()
        net = net.cuda()
        with torch.no_grad():
            output = net.forward(img[None], camera, mirror_camera, x_mirror=mirror_image[None])
        print('hi')

    def test_triplanenet_model(self):
        device = torch.device('cuda')
        triplane_net = TriplaneNetModel(torch.device('cpu'))
        c_poses = torch.cat([
            create_camera_for_yaw(-np.pi / 8, 2.7),
            create_camera_for_yaw(np.pi / 8, 2.7),
            create_camera_for_yaw(0, 2.7, height=0.2),
            create_camera_for_yaw(0, 2.7, height=-0.2),
        ]).to(device)

        name = "meta_fairy_tale_woman"
        real_label = np.load(f"D:/Projects/eg3d-preprocessor/test/dataset/c/{name}.npy")[None]
        real_img_512 = load_img(f"D:/Projects/eg3d-preprocessor/test/dataset/crop/{name}.jpg")
        # img = resize_img(real_img_512, 0.5)

        triplane_net.to(device)
        output = triplane_net.invert(real_img_512, real_label)
        images = triplane_net.render(output, c_poses)
        print('hi')