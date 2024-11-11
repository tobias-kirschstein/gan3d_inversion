from argparse import Namespace
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from dreifus.image import Img
from elias.util.io import resize_img

import gan3d_inversion
from gan3d_inversion.goae.configs.swin_config import get_config
from gan3d_inversion.goae.training.goae import Net


class GOAEModel:
    def __init__(self, device: torch.device = torch.device('cuda')):
        opts = Namespace(
            cfg=f'{str(Path(gan3d_inversion.goae.__file__).parent)}/configs/swinv2.yaml',
            G_ckpt=f'{Path.home()}/.cache/GOAE/ffhqrebalanced512-128.pkl',
            E_ckpt=f'{Path.home()}/.cache/GOAE/encoder_FFHQ.pt',
            AFA_ckpt=f'{Path.home()}/.cache/GOAE/afa_FFHQ.pt',

            mlp_layer=2,
            start_from_latent_avg=True,
        )

        ## build model
        swin_config = get_config(opts)
        self._net = Net(device, opts, swin_config)
        self._net.eval()

        self._device = device

    @torch.no_grad()
    def invert(self, image: np.ndarray, camera: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(camera.shape) == 1:
            camera = camera[None]
        device = self._device
        real_img = resize_img(image, 0.5)
        real_img_512 = torch.from_numpy(image).permute(2, 0, 1)[None]
        real_img = torch.from_numpy(real_img).permute(2, 0, 1)[None]

        real_img = torch.tensor(real_img).to(device).to(torch.float32) / 127.5 - 1.
        real_label = torch.tensor(camera, dtype=torch.float32).to(device)
        real_img_512 = torch.tensor(real_img_512).to(device).to(torch.float32) / 127.5 - 1.

        rec_img_dict, rec_img_dict_w = self._net(real_img, real_label, real_img_512, return_latents=True)

        mix_triplane = rec_img_dict['mix_triplane']
        rec_ws = rec_img_dict['rec_ws']

        return mix_triplane, rec_ws

    @torch.no_grad()
    def render(self, mix_triplane: torch.Tensor, rec_ws: torch.Tensor, cameras: torch.Tensor) -> List[np.ndarray]:
        B = len(cameras)
        rec_ws = rec_ws.repeat(B, 1, 1)
        mix_triplane = mix_triplane.repeat(B, 1, 1, 1, 1)
        img_dict_novel_view = self._net.decoder.synthesis(ws=rec_ws, c=cameras, triplane=mix_triplane, forward_triplane=True, noise_mode='const')
        img_novel_view = img_dict_novel_view["image"]

        images = [Img.from_normalized_torch(img).to_numpy().img for img in img_novel_view]
        return images
