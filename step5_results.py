import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def generate_report():
    # Load the original host and the watermarked version
    original = cv2.imread('dataset/host_256.png')
    watermarked = cv2.imread('results_watermarked.png')
    
    if original is None or watermarked is None:
        print("[ERROR] Could not find host_256.png or results_watermarked.png. Run evaluation first.")
        return
        
    # Load original logo and extracted logo
    orig_logo = cv2.imread('watermarks/logo_64.png', cv2.IMREAD_GRAYSCALE)
    extracted_logo = cv2.imread('results_extracted_logo.png', cv2.IMREAD_GRAYSCALE)

    if orig_logo is None or extracted_logo is None:
        print("[ERROR] Missing original or extracted logo files.")
        return

    # 1. Calculate PSNR & SSIM (Imperceptibility)
    p_score = psnr(original, watermarked, data_range=255)
    s_score = ssim(original, watermarked, channel_axis=2, data_range=255)

    # 2. Calculate NC (Robustness/Accuracy)
    # NC is the correlation between original and extracted watermark
    orig_norm = (orig_logo - np.mean(orig_logo)) / (np.std(orig_logo) + 1e-5)
    extr_norm = (extracted_logo - np.mean(extracted_logo)) / (np.std(extracted_logo) + 1e-5)
    nc_score = np.mean(orig_norm * extr_norm)

    print("\n" + "="*45)
    print("        FUTA M.TECH FINAL TEST REPORT       ")
    print("="*45)
    print(f"METRIC                   | VALUE")
    print(f"-------------------------|----------")
    print(f"PSNR (Imperceptibility)  | {p_score:.2f} dB")
    print(f"SSIM (Quality)           | {s_score:.4f}")
    print(f"NC (Extraction Accuracy) | {max(0, nc_score):.4f}")
    print(f"Data Split Used          | 80:10:10")
    print("="*45)
    print("\n[SUCCESS] Chapter 4 Data Generated.")

if __name__ == "__main__":
    generate_report()
