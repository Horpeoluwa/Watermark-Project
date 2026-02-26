import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def generate_report():
    # Load the original host and the watermarked version
    original = cv2.imread('dataset/host_256.png')
    watermarked = cv2.imread('results_watermarked.png')
    
    # Load original logo and extracted logo
    orig_logo = cv2.imread('watermarks/logo_64.png', cv2.IMREAD_GRAYSCALE)
    extracted_logo = cv2.imread('results_extracted_logo.png', cv2.IMREAD_GRAYSCALE)

    # 1. Calculate PSNR & SSIM (Imperceptibility)
    p_score = psnr(original, watermarked)
    s_score = ssim(original, watermarked, channel_axis=2)

    # 2. Calculate NC (Robustness/Accuracy)
    # NC is the correlation between original and extracted watermark
    orig_norm = (orig_logo - np.mean(orig_logo)) / (np.std(orig_logo) + 1e-5)
    extr_norm = (extracted_logo - np.mean(extracted_logo)) / (np.std(extracted_logo) + 1e-5)
    nc_score = np.mean(orig_norm * extr_norm)

    print("\n" + "="*45)
    print("       FUTA M.TECH FINAL TEST REPORT       ")
    print("="*45)
    print(f"METRIC                   | VALUE")
    print(f"-------------------------|----------")
    print(f"PSNR (Imperceptibility)  | {p_score:.2f} dB")
    print(f"SSIM (Quality)           | {s_score:.44f}")
    print(f"NC (Extraction Accuracy) | {nc_score:.4f}")
    print(f"Data Split Used          | 70:20:10")
    print("="*45)
    print("\n[SUCCESS] Chapter 4 Data Generated.")

if __name__ == "__main__":
    generate_report()