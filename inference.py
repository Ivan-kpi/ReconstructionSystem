import os
import warnings
import time
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import DATASETS, OUTPUT, SUPPORTED_EXTENSIONS, MODELS

from models.srcnn.models import SRCNN
from super_image import EdsrModel, ImageLoader
from basicsr.archs.rrdbnet_arch import RRDBNet as real_esrgan_RRDBNet
from models.esrgan.RRDBNet_arch import RRDBNet as esrgan_RRDBNet
from models.realesrgan.utils import RealESRGANer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def get_image_files(input_dir):
    return [f for f in os.listdir(input_dir) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

def run_srcnn():
    times = {}
    for scale, model_path in MODELS["SRCNN"].items():
        start_time = time.time()

        model = SRCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        for dataset_name, paths in DATASETS.items():
            input_dir = paths[f'LR_{scale}x']
            output_dir = os.path.join(OUTPUT, 'srcnn', dataset_name, f'x{scale}')
            os.makedirs(output_dir, exist_ok=True)

            filenames = get_image_files(input_dir)
            for filename in tqdm(filenames, desc=f'SRCNN {dataset_name} [x{scale}]'):
                img = Image.open(os.path.join(input_dir, filename)).convert('RGB')
                img = img.resize((img.width * scale, img.height * scale), resample=Image.BICUBIC)

                ycbcr = img.convert('YCbCr')
                y, cb, cr = ycbcr.split()
                y_np = np.array(y).astype(np.float32) / 255.0
                y_tensor = torch.from_numpy(y_np).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    out_tensor = model(y_tensor).clamp(0.0, 1.0)

                out_y = out_tensor.squeeze().cpu().numpy()
                out_y_img = Image.fromarray((out_y * 255.0).round().astype(np.uint8))
                out_img = Image.merge('YCbCr', [out_y_img, cb, cr]).convert('RGB')

                out_img.save(os.path.join(output_dir, filename))

        end_time = time.time() - start_time
        print(f'\nSRCNN [x{scale}]: {end_time:.2f} секунд')
    return  times

def run_edsr():
    times = {}
    for scale, model_path in MODELS["EDSR"].items():
        start_time = time.time()

        model = EdsrModel.from_pretrained(model_path, scale=scale).to(device)

        for dataset_name, paths in DATASETS.items():
            input_dir = paths[f'LR_{scale}x']
            output_dir = os.path.join(OUTPUT, 'edsr', dataset_name, f'x{scale}')
            os.makedirs(output_dir, exist_ok=True)

            filenames = get_image_files(input_dir)

            for filename in tqdm(filenames, desc=f'EDSR {dataset_name} [x{scale}]'):
                img = Image.open(os.path.join(input_dir, filename)).convert('RGB')
                inputs = ImageLoader.load_image(img).to(device)

                with torch.no_grad():
                    preds = model(inputs).clamp(0.0, 1.0)

                sr_np = preds.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255.0
                sr_img = Image.fromarray(np.uint8(sr_np.clip(0, 255)), mode="RGB")
                sr_img.save(os.path.join(output_dir, filename))

        end_time = time.time() - start_time
        print(f'\nEDSR [x{scale}]: {end_time:.2f} секунд')
    return times

def run_esrgan():
    times = {}
    for scale, model_path in MODELS["ESRGAN"].items():
        start_time = time.time()

        model = esrgan_RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval().to(device)

        for dataset_name, paths in DATASETS.items():

            input_dir = paths[f'LR_{scale}x']
            output_dir = os.path.join(OUTPUT, 'esrgan', dataset_name, f'x{scale}')
            os.makedirs(output_dir, exist_ok=True)

            filenames = get_image_files(input_dir)
            for filename in tqdm(filenames, desc=f'ESRGAN {dataset_name} [x{scale}]'):

                img = Image.open(os.path.join(input_dir, filename)).convert('RGB')
                img_np = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

                with torch.no_grad():
                    out_tensor = model(img_tensor).squeeze().cpu().clamp_(0, 1).numpy()

                out_np = (out_tensor.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
                out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, filename), out_bgr)

        end_time = time.time() - start_time
        print(f'\nESRGAN [x{scale}]: {end_time:.2f} секунд')
    return times

def run_real_esrgan():
    for scale, model_path in MODELS["Real-ESRGAN"].items():
        start_time = time.time()

        model = real_esrgan_RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        upscaler = RealESRGANer(scale=scale, model_path=model_path, model=model, tile=0, tile_pad=10,
                                 pre_pad=0, half=(device.type == 'cuda'), device=device)

        for dataset_name, paths in DATASETS.items():
            input_dir = paths[f'LR_{scale}x']
            output_dir = os.path.join(OUTPUT, 'real-esrgan', dataset_name, f'x{scale}')
            os.makedirs(output_dir, exist_ok=True)

            filenames = get_image_files(input_dir)
            for filename in tqdm(filenames, desc=f'Real-esrgan {dataset_name} [x{scale}]'):
                img = cv2.imread(os.path.join(input_dir, filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                try:
                    output, _ = upscaler.enhance(img, outscale=scale)
                except Exception as e:
                    print(f"Помилка обробки {filename}: {e}")
                    continue
                output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, filename), output_bgr)

        end_time = time.time() - start_time
        print(f'\nReal-ESRGAN [x{scale}]: {end_time:.2f} секунд')

if __name__ == "__main__":

    run_srcnn()

    run_edsr()

    run_esrgan()

    run_real_esrgan()


