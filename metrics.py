import os
import warnings
import cv2
import torch
import numpy as np
import lpips
import pandas as pd
import piq
from tqdm import tqdm
from basicsr.metrics.niqe import calculate_niqe
from config import DATASETS, SCALES, MODELS, METRICS_PATH, OUTPUT
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

os.makedirs(METRICS_PATH, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_model = lpips.LPIPS(net='alex').to(device)

def calculate_metrics(hr_img, sr_img):
    h = min(hr_img.shape[0], sr_img.shape[0])
    w = min(hr_img.shape[1], sr_img.shape[1])

    hr_cropped = hr_img[:h, :w]
    sr_cropped = sr_img[:h, :w]

    # PSNR працює напряму по numpy [0,255]
    psnr = compare_psnr(hr_cropped, sr_cropped, data_range=255)

    # NIQE — окремо по numpy (з Y-каналу)
    niqe_score = calculate_niqe(sr_cropped, crop_border=0, input_order='HWC', convert_to='y')

    # Перетворення в тензори [0, 1]
    hr_tensor = torch.from_numpy(hr_cropped).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    sr_tensor = torch.from_numpy(sr_cropped).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    # LPIPS очікує [-1, 1], тому просто: *2 - 1
    lpips_score = lpips_model(hr_tensor * 2 - 1, sr_tensor * 2 - 1).item()

    # MS-SSIM працює на [0, 1]
    ms_ssim_score = piq.multi_scale_ssim(hr_tensor, sr_tensor, data_range=1.0).item()

    return psnr, lpips_score, ms_ssim_score, niqe_score


results = []

for model in MODELS.keys():
    model_dir = os.path.join(OUTPUT, model.lower())
    if not os.path.exists(model_dir):
        print(f"\nМодель {model} пропущено (директорія відсутня)")
        continue

    for dataset_name, dataset_paths in DATASETS.items():
        dataset_dir = os.path.join(model_dir, dataset_name)
        if not os.path.exists(dataset_dir):
            print(f"\nДатасет {dataset_name} для {model} пропущено (директорія відсутня)")
            continue

        for scale in SCALES:
            scale_dir = os.path.join(dataset_dir, scale)
            if not os.path.exists(scale_dir):
                print(f"\nМасштаб {scale} для {model} / {dataset_name} пропущено (директорія відсутня)")
                continue

            hr_dir = dataset_paths['HR']
            filenames = [f for f in os.listdir(hr_dir) if f.endswith('.png')]

            for filename in tqdm(filenames, desc=f'{model} [{dataset_name}] {scale}'):
                hr_path = os.path.join(hr_dir, filename)
                sr_path = os.path.join(scale_dir, filename)

                if not os.path.exists(sr_path):
                    print(f"\nФайл не знайдено: {sr_path}")
                    continue

                hr_img = cv2.imread(hr_path)
                sr_img = cv2.imread(sr_path)

                if hr_img is None or sr_img is None:
                    print(f"\nПроблема з файлом: {filename}")
                    continue

                psnr, lpips_score, ms_ssim_score, niqe_score = calculate_metrics(hr_img, sr_img)
                results.append({
                    'Filename': filename,
                    'Model': model,
                    'Dataset': dataset_name,
                    'Scale': scale,
                    'PSNR': psnr,
                    'LPIPS': lpips_score,
                    'MS_SSIM': ms_ssim_score,
                    'NIQE': niqe_score
                })


if results:
    df = pd.DataFrame(results)
    output_path = os.path.join(METRICS_PATH, "metrics.csv")
    df.to_csv(output_path, index=False)
    print("\nЗбережено metrics.csv")
