from unittest import TestCase

import numpy as np
import torch
from elias.util import load_img

from gan3d_inversion.goae.camera_utils import create_camera_for_yaw
from gan3d_inversion.goae.goae_model import GOAEModel


class GOAETest(TestCase):

    def test_goae(self):
        device = torch.device('cuda')
        goae_model = GOAEModel()

        name = "meta_fairy_tale_woman"
        real_label = np.load(f"D:/Projects/eg3d-preprocessor/test/dataset/c/{name}.npy")[None]
        real_img_512 = load_img(f"D:/Projects/eg3d-preprocessor/test/dataset/crop/{name}.jpg")
        mix_triplane, rec_ws = goae_model.invert(real_img_512, real_label)

        c_poses = torch.cat([
            create_camera_for_yaw(-np.pi / 8, 2.7),
            create_camera_for_yaw(np.pi / 8, 2.7),
            create_camera_for_yaw(0, 2.7, height=0.2),
            create_camera_for_yaw(0, 2.7, height=-0.2),
        ]).to(device)
        renderings = goae_model.render(mix_triplane, rec_ws, c_poses)
