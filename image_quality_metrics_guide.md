# Image Quality Metrics: Complete Evaluation Guide

## Quick Overview

Here's how to evaluate **PSNR**, **SSIM**, **LPIPS**, and **FID** scores for image quality assessment:

| Metric | Range | Higher is Better | Use Case | Typical Good Values |
|--------|-------|-----------------|----------|-------------------|
| **PSNR** | 0 to ∞ dB | ✅ Yes | Pixel-level reconstruction | 25-40 dB |
| **SSIM** | -1 to 1 | ✅ Yes | Structural similarity | 0.8-0.99 |
| **LPIPS** | 0 to ∞ | ❌ No (lower better) | Perceptual similarity | 0.0-0.3 |
| **FID** | 0 to ∞ | ❌ No (lower better) | Distribution similarity | 1-50 |

## 🔧 Installation

```bash
# Core dependencies
pip install torch torchvision pillow numpy scipy

# For LPIPS
pip install lpips

# For FID
pip install pytorch-fid

# Alternative: All at once
pip install torch torchvision pillow numpy scipy lpips pytorch-fid
```

## 📊 1. PSNR (Peak Signal-to-Noise Ratio)

**What it measures**: Pixel-level reconstruction accuracy  
**Best for**: Image restoration, compression, super-resolution

### Basic Usage

```python
import torch
import torch.nn.functional as F

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate PSNR between two images
    
    Args:
        img1, img2: torch.Tensor with shape (C, H, W) or (B, C, H, W)
        max_val: 1.0 for normalized images, 255 for uint8
    
    Returns:
        PSNR in dB (higher is better)
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Perfect reconstruction
    
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

# Example usage
original = torch.rand(3, 256, 256)  # Original image
reconstructed = original + torch.randn_like(original) * 0.1  # Add noise

psnr_score = calculate_psnr(original, reconstructed)
print(f"PSNR: {psnr_score:.2f} dB")
```

### Batch Processing

```python
def batch_psnr(images1, images2, max_val=1.0):
    """Calculate PSNR for multiple image pairs"""
    batch_size = images1.shape[0]
    psnr_values = []
    
    for i in range(batch_size):
        psnr = calculate_psnr(images1[i], images2[i], max_val)
        psnr_values.append(psnr)
    
    return psnr_values

# Example with batch
batch_original = torch.rand(10, 3, 256, 256)
batch_reconstructed = batch_original + torch.randn_like(batch_original) * 0.1

psnr_list = batch_psnr(batch_original, batch_reconstructed)
print(f"Average PSNR: {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} dB")
```

### Interpretation

```python
def interpret_psnr(psnr_value):
    """Interpret PSNR score"""
    if psnr_value > 40:
        return "Excellent quality"
    elif psnr_value > 30:
        return "Good quality"
    elif psnr_value > 20:
        return "Acceptable quality"
    else:
        return "Poor quality"

print(f"Quality: {interpret_psnr(psnr_score)}")
```

## 📏 2. SSIM (Structural Similarity Index)

**What it measures**: Structural and perceptual similarity  
**Best for**: Human visual perception correlation

### Implementation

```python
import torch.nn.functional as F

def calculate_ssim(img1, img2, window_size=11, max_val=1.0):
    """
    Calculate SSIM between two images
    
    Args:
        img1, img2: torch.Tensor (C, H, W) 
        window_size: Gaussian window size (default 11)
        max_val: Maximum pixel value
    
    Returns:
        SSIM value between -1 and 1 (1 is perfect)
    """
    def gaussian_window(window_size, sigma=1.5):
        coords = torch.arange(window_size, dtype=torch.float)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.unsqueeze(1) * g.unsqueeze(0)
    
    # Add batch dimension if needed
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    _, channels, height, width = img1.shape
    
    # Create Gaussian window
    window = gaussian_window(window_size).repeat(channels, 1, 1, 1)
    
    # Calculate local means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate local variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu1_mu2
    
    # SSIM constants
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    # Calculate SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator
    
    return ssim_map.mean().item()

# Example usage
ssim_score = calculate_ssim(original, reconstructed)
print(f"SSIM: {ssim_score:.4f}")
```

### Using scikit-image (Alternative)

```python
from skimage.metrics import structural_similarity as ssim
import numpy as np

def calculate_ssim_skimage(img1, img2):
    """Calculate SSIM using scikit-image"""
    # Convert tensors to numpy
    if isinstance(img1, torch.Tensor):
        img1 = img1.permute(1, 2, 0).numpy()  # CHW -> HWC
        img2 = img2.permute(1, 2, 0).numpy()
    
    # Calculate SSIM
    ssim_value = ssim(img1, img2, multichannel=True, data_range=1.0)
    return ssim_value

# Example
ssim_score = calculate_ssim_skimage(original, reconstructed)
print(f"SSIM (scikit-image): {ssim_score:.4f}")
```

## 🧠 3. LPIPS (Learned Perceptual Image Patch Similarity)

**What it measures**: Perceptual similarity using deep features  
**Best for**: Human perceptual judgment correlation

### Setup and Usage

```python
import lpips

class LPIPSEvaluator:
    def __init__(self, net='alex', device='cuda'):
        """
        Initialize LPIPS evaluator
        
        Args:
            net: 'alex', 'vgg', or 'squeeze'
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.lpips_fn = lpips.LPIPS(net=net).to(device)
        self.lpips_fn.eval()
    
    def calculate_lpips(self, img1, img2):
        """
        Calculate LPIPS distance
        
        Args:
            img1, img2: torch.Tensor (C, H, W) or (B, C, H, W)
                       Values should be in [-1, 1] range!
        
        Returns:
            LPIPS distance (lower is better)
        """
        # Ensure batch dimension
        if len(img1.shape) == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        
        # Move to device and ensure [-1, 1] range
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        with torch.no_grad():
            lpips_distance = self.lpips_fn(img1, img2)
        
        return lpips_distance.item()

# Example usage
lpips_evaluator = LPIPSEvaluator()

# Convert from [0, 1] to [-1, 1] range
img1_normalized = original * 2 - 1
img2_normalized = reconstructed * 2 - 1

lpips_score = lpips_evaluator.calculate_lpips(img1_normalized, img2_normalized)
print(f"LPIPS: {lpips_score:.4f}")
```

### Quick LPIPS Calculation

```python
def quick_lpips(img1, img2, net='alex'):
    """Quick LPIPS calculation"""
    lpips_fn = lpips.LPIPS(net=net)
    
    # Ensure [-1, 1] range
    if img1.max() <= 1.0:  # Assuming [0, 1] input
        img1 = img1 * 2 - 1
        img2 = img2 * 2 - 1
    
    # Add batch dimension if needed
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    with torch.no_grad():
        distance = lpips_fn(img1, img2)
    
    return distance.item()

# Example
lpips_score = quick_lpips(original, reconstructed)
print(f"LPIPS: {lpips_score:.4f}")
```

## 📈 4. FID (Fréchet Inception Distance)

**What it measures**: Distribution similarity between real and generated images  
**Best for**: Evaluating generative models

### Using pytorch-fid Library

```bash
# Command line usage
pip install pytorch-fid

# Calculate FID between two folders
python -m pytorch_fid path/to/real/images path/to/generated/images --device cuda
```

### Custom Implementation

```python
import numpy as np
from scipy import linalg
from torchvision import transforms
from pytorch_fid.inception import InceptionV3

class FIDEvaluator:
    def __init__(self, device='cuda', dims=2048):
        """
        Initialize FID evaluator
        
        Args:
            device: 'cuda' or 'cpu'
            dims: Inception feature dimensions (2048 default)
        """
        self.device = device
        self.dims = dims
        
        # Load Inception model
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.inception = InceptionV3([block_idx]).to(device)
        self.inception.eval()
    
    def extract_features(self, images):
        """Extract Inception features"""
        features_list = []
        batch_size = 50
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(self.device)
                
                # Resize to 299x299 for Inception
                if batch.shape[-1] != 299:
                    batch = F.interpolate(batch, size=(299, 299), 
                                        mode='bilinear', align_corners=False)
                
                features = self.inception(batch)[0]
                
                # Global average pooling
                if features.shape[2] != 1 or features.shape[3] != 1:
                    features = F.adaptive_avg_pool2d(features, (1, 1))
                
                features = features.cpu().numpy().reshape(features.shape[0], -1)
                features_list.append(features)
        
        return np.concatenate(features_list, axis=0)
    
    def calculate_fid(self, real_images, generated_images):
        """
        Calculate FID score
        
        Args:
            real_images, generated_images: torch.Tensor (B, C, H, W) in [0, 1]
        
        Returns:
            FID score (lower is better)
        """
        # Extract features
        real_features = self.extract_features(real_images)
        gen_features = self.extract_features(generated_images)
        
        # Calculate statistics
        mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
        
        # Calculate FID
        fid = self.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        """Calculate Fréchet distance"""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        
        # Calculate sqrt of product of covariances
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Handle numerical errors
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                raise ValueError("Imaginary component in covariance mean")
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

# Example usage
fid_evaluator = FIDEvaluator()

# Generate dummy data (replace with your real and generated images)
real_images = torch.rand(100, 3, 256, 256)
generated_images = torch.rand(100, 3, 256, 256)

fid_score = fid_evaluator.calculate_fid(real_images, generated_images)
print(f"FID: {fid_score:.2f}")
```

## 🔄 Complete Evaluation Pipeline

Here's a comprehensive evaluation class that combines all metrics:

```python
class ImageQualityEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        self.lpips_evaluator = LPIPSEvaluator(device=device)
        self.fid_evaluator = FIDEvaluator(device=device)
    
    def evaluate_reconstruction(self, original, reconstructed, max_val=1.0):
        """Evaluate reconstruction quality"""
        results = {}
        
        # PSNR
        psnr_values = [calculate_psnr(original[i], reconstructed[i], max_val) 
                      for i in range(len(original))]
        results['psnr'] = {
            'mean': np.mean(psnr_values),
            'std': np.std(psnr_values),
            'values': psnr_values
        }
        
        # SSIM
        ssim_values = [calculate_ssim(original[i], reconstructed[i], max_val=max_val) 
                      for i in range(len(original))]
        results['ssim'] = {
            'mean': np.mean(ssim_values),
            'std': np.std(ssim_values),
            'values': ssim_values
        }
        
        # LPIPS
        orig_lpips = original * 2 - 1 if max_val == 1.0 else (original / max_val) * 2 - 1
        recon_lpips = reconstructed * 2 - 1 if max_val == 1.0 else (reconstructed / max_val) * 2 - 1
        
        lpips_values = [self.lpips_evaluator.calculate_lpips(orig_lpips[i], recon_lpips[i]) 
                       for i in range(len(original))]
        results['lpips'] = {
            'mean': np.mean(lpips_values),
            'std': np.std(lpips_values),
            'values': lpips_values
        }
        
        return results
    
    def evaluate_generation(self, real_images, generated_images):
        """Evaluate generation quality"""
        results = {}
        
        # FID
        fid_score = self.fid_evaluator.calculate_fid(real_images, generated_images)
        results['fid'] = fid_score
        
        return results

# Usage example
evaluator = ImageQualityEvaluator()

# For reconstruction tasks
reconstruction_results = evaluator.evaluate_reconstruction(original_images, reconstructed_images)

# For generation tasks
generation_results = evaluator.evaluate_generation(real_images, generated_images)

print("Reconstruction Results:")
for metric, values in reconstruction_results.items():
    if isinstance(values, dict):
        print(f"{metric.upper()}: {values['mean']:.4f} ± {values['std']:.4f}")
    else:
        print(f"{metric.upper()}: {values:.4f}")

print("\nGeneration Results:")
for metric, value in generation_results.items():
    print(f"{metric.upper()}: {value:.4f}")
```

## 📁 Evaluating Images from Folders

```python
from PIL import Image
import os

def load_images_from_folder(folder_path, max_images=None, size=(256, 256)):
    """Load images from folder"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    
    images = []
    image_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if max_images:
        image_files = image_files[:max_images]
    
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        images.append(img_tensor)
    
    return torch.stack(images)

# Example usage
original_folder = "path/to/original/images"
reconstructed_folder = "path/to/reconstructed/images"

original_images = load_images_from_folder(original_folder, max_images=100)
reconstructed_images = load_images_from_folder(reconstructed_folder, max_images=100)

evaluator = ImageQualityEvaluator()
results = evaluator.evaluate_reconstruction(original_images, reconstructed_images)
```

## 📊 Interpretation Guidelines

### PSNR Interpretation
```python
def interpret_psnr(psnr):
    if psnr > 40: return "Excellent (40+ dB)"
    elif psnr > 30: return "Good (30-40 dB)"
    elif psnr > 20: return "Acceptable (20-30 dB)"
    else: return "Poor (<20 dB)"
```

### SSIM Interpretation
```python
def interpret_ssim(ssim):
    if ssim > 0.95: return "Excellent (0.95+)"
    elif ssim > 0.8: return "Good (0.8-0.95)"
    elif ssim > 0.6: return "Acceptable (0.6-0.8)"
    else: return "Poor (<0.6)"
```

### LPIPS Interpretation
```python
def interpret_lpips(lpips):
    if lpips < 0.1: return "Excellent (<0.1)"
    elif lpips < 0.3: return "Good (0.1-0.3)"
    elif lpips < 0.6: return "Acceptable (0.3-0.6)"
    else: return "Poor (>0.6)"
```

### FID Interpretation
```python
def interpret_fid(fid):
    if fid < 10: return "Excellent (<10)"
    elif fid < 50: return "Good (10-50)"
    elif fid < 100: return "Acceptable (50-100)"
    else: return "Poor (>100)"
```

## 🎯 When to Use Each Metric

### Use PSNR when:
- ✅ Evaluating pixel-level reconstruction accuracy
- ✅ Comparing compression algorithms
- ✅ You need a simple, fast metric
- ❌ Perceptual quality matters more than pixel accuracy

### Use SSIM when:
- ✅ Evaluating structural similarity
- ✅ Perceptual quality is important
- ✅ Comparing image enhancement methods
- ❌ You need distribution-level evaluation

### Use LPIPS when:
- ✅ Human perceptual judgment is the gold standard
- ✅ Evaluating style transfer or image translation
- ✅ Pixel-level metrics don't correlate with perception
- ❌ Computational resources are very limited

### Use FID when:
- ✅ Evaluating generative models (GANs, diffusion models)
- ✅ Comparing distributions of image sets
- ✅ You have sufficient data (>1000 images recommended)
- ❌ Evaluating single image pairs

## 🚀 Quick Start Template

```python
import torch
import numpy as np

# Load your images here
original_images = torch.rand(10, 3, 256, 256)  # Replace with your data
reconstructed_images = torch.rand(10, 3, 256, 256)  # Replace with your data

# Initialize evaluator
evaluator = ImageQualityEvaluator()

# Evaluate
results = evaluator.evaluate_reconstruction(original_images, reconstructed_images)

# Print results
print("📊 Image Quality Assessment Results")
print("=" * 40)
for metric, values in results.items():
    if isinstance(values, dict):
        print(f"{metric.upper():>6}: {values['mean']:.4f} ± {values['std']:.4f}")
    else:
        print(f"{metric.upper():>6}: {values:.4f}")

# Interpret results
print("\n🎯 Quality Assessment:")
print(f"PSNR: {interpret_psnr(results['psnr']['mean'])}")
print(f"SSIM: {interpret_ssim(results['ssim']['mean'])}")
print(f"LPIPS: {interpret_lpips(results['lpips']['mean'])}")
```

---

This guide provides comprehensive coverage of all four metrics with practical, ready-to-use code examples. Choose the metrics that best fit your specific use case and requirements!