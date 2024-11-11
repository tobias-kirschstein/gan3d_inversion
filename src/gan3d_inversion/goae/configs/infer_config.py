import os
import argparse
from pathlib import Path

import gan3d_inversion


def get_parser():
    parser = argparse.ArgumentParser()

    ## path 
    parser.add_argument('--cfg', type=str,  metavar="FILE", help='path to config file', default=f'{str(Path(gan3d_inversion.goae.__file__).parent)}/configs/swinv2.yaml')
    parser.add_argument("--data", type=str, help='path to data directory', default='../example/real_person')
    parser.add_argument("--G_ckpt", type=str, help='path to generator model', default=f'{Path.home()}/.cache/GOAE/ffhqrebalanced512-128.pkl')
    parser.add_argument("--E_ckpt", type=str, help='path to GOAE encoder checkpoint', default=f'{Path.home()}/.cache/GOAE/encoder_FFHQ.pt')
    parser.add_argument("--AFA_ckpt", type=str, help='path to AFA model checkpoint', default=f'{Path.home()}/.cache/GOAE/afa_FFHQ.pt')
    parser.add_argument("--outdir", type=str, help='path to output directory', default='../output/')
    parser.add_argument("--cuda", type=str, help="specify used cuda idx ", default='0')

    ## model 
    parser.add_argument("--mlp_layer", type=int, default=2)
    parser.add_argument("--start_from_latent_avg", type=bool, default=True)

    ## other
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--w_frames', type=int, default=240)
    parser.add_argument("--multi_view", action="store_true", default=False)
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--shape", action="store_true", default=False)
    parser.add_argument("--edit", action="store_true", default=False)

    ## edit 
    parser.add_argument("--edit_attr", type=str, help="editing attribute direction", default="glass")
    parser.add_argument("--alpha", type=float, help="editing alpha", default=1.0)

    return parser