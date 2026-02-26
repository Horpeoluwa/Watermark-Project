import torch
import torch.nn as nn

class WatermarkEncoder(nn.Module):
    def __init__(self):
        super(WatermarkEncoder, self).__init__()
        # This part takes the 3-channel image + 1-channel watermark = 4 channels
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1), # Output 3 channels (RGB image)
            nn.Tanh() # Keeps pixels in a valid range
        )

    def forward(self, image, watermark):
        # We need to make the watermark the same size as the image to combine them
        watermark_upsampled = torch.nn.functional.interpolate(watermark, size=(256, 256))
        # Concatenate them: Putting them together
        x = torch.cat([image, watermark_upsampled], dim=1)
        return self.encoder(x)

class WatermarkDecoder(nn.Module):
    def __init__(self):
        super(WatermarkDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1), # Output 1 channel (Grayscale logo)
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.decoder(x)
        # Resize back to 64x64
        return torch.nn.functional.interpolate(x, size=(64, 64))