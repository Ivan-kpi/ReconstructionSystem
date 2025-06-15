import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff')

DATASETS = {
    "DIV2K": {
        "HR": os.path.join(PROJECT_ROOT, 'input', 'DIV2K', 'DIV2K_valid_HR'),
        "LR_2x": os.path.join(PROJECT_ROOT, 'input', 'DIV2K', 'DIV2K_valid_LR_x2'),
        "LR_4x": os.path.join(PROJECT_ROOT, 'input', 'DIV2K', 'DIV2K_valid_LR_x4'),
    },
    "URBAN100": {
        "HR": os.path.join(PROJECT_ROOT, 'input', 'URBAN100', 'HIGH x4 URban100'),
        "LR_2x": os.path.join(PROJECT_ROOT, 'input', 'URBAN100', 'LOW x2 URban100'),
        "LR_4x": os.path.join(PROJECT_ROOT, 'input', 'URBAN100', 'LOW x4 URban100'),
    },
    "BSD100": {
        "HR": os.path.join(PROJECT_ROOT, 'input', 'BSD100', 'HR'),
        "LR_2x": os.path.join(PROJECT_ROOT, 'input', 'BSD100', 'LR_2x'),
        "LR_4x": os.path.join(PROJECT_ROOT, 'input', 'BSD100', 'LR_4x'),
    }
}

SCALES = ['x2', 'x4']

MODELS = {
    "SRCNN": {
        2: "models/srcnn/model/SRCNN_x2.pth",
        4: "models/srcnn/model/SRCNN_x4.pth"
    },

    "EDSR" : {
        2: "eugenesiow/edsr-base",
        4: "eugenesiow/edsr-base"
    },

    "ESRGAN" : {
        4: "models/esrgan/RRDB_ESRGAN_x4.pth"
    },

    "Real-ESRGAN" : {
        2: "models/realesrgan/model/RealESRGAN_x2plus.pth",
        4: "models/realesrgan/model/RealESRGAN_x4plus.pth"
    }
}

METRICS = ['PSNR', 'LPIPS', 'MS_SSIM', 'NIQE']

OUTPUT = os.path.join(PROJECT_ROOT, 'output')

METRICS_PATH = os.path.join(PROJECT_ROOT, 'metrics')

PLOTS_PATH = os.path.join(PROJECT_ROOT, 'plots')