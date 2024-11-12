from pathlib import Path

dataset_paths = {
    'test': '',
    'train': '',
}

model_paths = {
    'eg3d_ffhq': f"{Path.home()}/.cache/TriplaneNet/ffhq512-128.pth",
    'ir_se50': f"{Path.home()}/.cache/TriplaneNet/model_ir_se50.pth",
    'circular_face': f"{Path.home()}/.cache/TriplaneNet/CurricularFace_Backbone.pth",
    'mtcnn_pnet': f"{Path.home()}/.cache/TriplaneNet/mtcnn/pnet.npy",
    'mtcnn_rnet': f"{Path.home()}/.cache/TriplaneNet/mtcnn/rnet.npy",
    'mtcnn_onet': f"{Path.home()}/.cache/TriplaneNet/mtcnn/onet.npy",
    'discriminator': f"{Path.home()}/.cache/TriplaneNet/discriminator.pth"
}
