import cv2
import os
import numpy as np

def preprocess():
    # Define paths
    host_path = 'dataset/host.jpg'
    logo_path = 'watermarks/logo.png'
    
    # 1. Check if files exist
    if not os.path.exists(host_path):
        print("Error: 'host.jpg' not found in dataset folder!")
        return
    if not os.path.exists(logo_path):
        print("Error: 'logo.png' not found in watermarks folder!")
        return

    # 2. Process Host Image (Resize to 256x256)
    host = cv2.imread(host_path)
    host_256 = cv2.resize(host, (256, 256))
    cv2.imwrite('dataset/host_256.png', host_256)
    
    # 3. Process Watermark (Resize to 64x64 and Grayscale)
    # Your proposal mentions grayscale for the watermark
    logo = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
    logo_64 = cv2.resize(logo, (64, 64))
    cv2.imwrite('watermarks/logo_64.png', logo_64)
    
    print("\n" + "="*40)
    print("STEP 2: PREPROCESSING COMPLETE")
    print(f"Host Image: 256x256 saved to 'dataset/host_256.png'")
    print(f"Watermark: 64x64 saved to 'watermarks/logo_64.png'")
    print("="*40 + "\n")

if __name__ == "__main__":
    preprocess()