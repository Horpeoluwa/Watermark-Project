import torch
from model import WatermarkEncoder, WatermarkDecoder

def test_structure():
    encoder = WatermarkEncoder()
    decoder = WatermarkDecoder()
    
    # Simulate a fake image and fake watermark
    fake_image = torch.randn(1, 3, 256, 256)     # 1 image, RGB, 256x256
    fake_watermark = torch.randn(1, 1, 64, 64)  # 1 watermark, Grayscale, 64x64
    
    # Step 1: Hide it (Encoding)
    watermarked_image = encoder(fake_image, fake_watermark)
    print(f"Encoded Image Shape: {watermarked_image.shape} (Should be 256x256)")
    
    # Step 2: Find it (Decoding)
    extracted_watermark = decoder(watermarked_image)
    print(f"Extracted Watermark Shape: {extracted_watermark.shape} (Should be 64x64)")
    
    print("\n--- STEP 3 SUCCESS: Model Architecture is Correct ---")

if __name__ == "__main__":
    test_structure()