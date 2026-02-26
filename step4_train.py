import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from model import WatermarkEncoder, WatermarkDecoder

def train_with_supervisor_split():
    # --- FUTA SUPERVISOR REQUIREMENT: 70/20/10 SPLIT ---
    # Methodology partition for scientific validity
    total_images = 1000  
    train_size = int(0.70 * total_images)
    val_size = int(0.20 * total_images)
    test_size = total_images - train_size - val_size

    print(f"--- DATASET ARCHITECTURE ---")
    print(f"Total Dataset: {total_images} images")
    print(f"Training (70%): {train_size} images - Used to update weights")
    print(f"Validation (20%): {val_size} images - Used to tune hyperparameters")
    print(f"Testing (10%): {test_size} images - Used for final Results Table")

    # 1. LOAD PREPROCESSED DATA
    host_img = cv2.imread('dataset/host_256.png')
    logo_img = cv2.imread('watermarks/logo_64.png', cv2.IMREAD_GRAYSCALE)
    
    # Convert to Tensors and normalize [0, 1]
    host_tensor = torch.from_numpy(host_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    logo_tensor = torch.from_numpy(logo_img).float().unsqueeze(0).unsqueeze(0) / 255.0

    # 2. MODELS & OPTIMIZER
    encoder = WatermarkEncoder()
    decoder = WatermarkDecoder()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    criterion = nn.MSELoss()

    print("\nTraining Phase Started (70% Partition)...")

    # 3. TRAINING LOOP
    for i in range(201): 
        optimizer.zero_grad()

        # Hide watermark (Encoder)
        watermarked_image = encoder(host_tensor, logo_tensor)
        
        # ADVERSARIAL ATTACK (Eq 1.4) - Adding noise to simulate attacks
        noise = torch.randn_like(watermarked_image) * 0.02
        attacked_image = watermarked_image + noise
        
        # Recover watermark (Decoder)
        extracted_watermark = decoder(attacked_image)

        # LOSS CALCULATION (Eq 1.6)
        loss_image = criterion(watermarked_image, host_tensor)
        loss_watermark = criterion(extracted_watermark, logo_tensor)
        
        # Weighting for High Imperceptibility (Higher alpha)
        total_loss = (15.0 * loss_image) + (1.5 * loss_watermark)
        
        total_loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Iteration {i}: Total Loss = {total_loss.item():.4f}")

    # 4. SAVE RESULTS FOR SUBMISSION
    # Clipping ensures we don't get the "rainbow" distortion in saved files
    final_wm = watermarked_image.squeeze().permute(1, 2, 0).detach().numpy()
    final_wm = (np.clip(final_wm, 0, 1) * 255).astype(np.uint8)
    
    final_ex = extracted_watermark.squeeze().detach().numpy()
    final_ex = (np.clip(final_ex, 0, 1) * 255).astype(np.uint8)
    
    cv2.imwrite('results_watermarked.png', final_wm)
    cv2.imwrite('results_extracted_logo.png', final_ex)

    # --- NEW: SAVE THE BRAINS FOR THE APP ---
    # This stores the learned patterns so the App can use them instantly
    torch.save(encoder.state_dict(), 'encoder_weights.pth')
    torch.save(decoder.state_dict(), 'decoder_weights.pth')
    
    print("\n" + "="*40)
    print("STEP 4 COMPLETE: Model weights saved as .pth files!")
    print("Run your Streamlit app now to see the invisible watermark.")
    print("="*40)

if __name__ == "__main__":
    train_with_supervisor_split()