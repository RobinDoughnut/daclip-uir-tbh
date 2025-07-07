#!/usr/bin/env python3
"""
Image Quality Metrics Evaluation Script
Comprehensive guide for computing PSNR, SSIM, LPIPS, and FID scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
from typing import Union, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# DEPENDENCIES AND IMPORTS
# ============================================================================

def install_requirements():
    """Install required packages if not available"""
    required_packages = [
        'torch>=1.9.0',
        'torchvision>=0.10.0', 
        'pillow>=8.0.0',
        'scipy>=1.7.0',
        'lpips>=0.1.4',
        'pytorch-fid>=0.3.0'
    ]
    
    print("Required packages:")
    for pkg in required_packages:
        print(f"  - {pkg}")
    print("\nInstall with: pip install torch torchvision pillow scipy lpips pytorch-fid")

try:
    import lpips
    import scipy.linalg
    from torchvision import transforms, models
    from pytorch_fid import fid_score
    from pytorch_fid.inception import InceptionV3
except ImportError as e:
    print(f"Missing dependency: {e}")
    install_requirements()
    exit(1)

# ============================================================================
# PSNR (Peak Signal-to-Noise Ratio)
# ============================================================================

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calculate PSNR between two images.
    
    Args:
        img1, img2: Images as tensors with shape (B, C, H, W) or (C, H, W)
        max_val: Maximum possible pixel value (1.0 for normalized, 255 for uint8)
    
    Returns:
        PSNR value in dB (higher is better)
    
    Range: 0 to ∞ (typically 20-40 dB for good quality)
    Use case: Pixel-level reconstruction accuracy
    """
    # Ensure same shape
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"
    
    # Calculate MSE
    mse = torch.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')  # Perfect reconstruction
    
    # Calculate PSNR
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def batch_psnr(images1: torch.Tensor, images2: torch.Tensor, max_val: float = 1.0) -> List[float]:
    """Calculate PSNR for batch of images"""
    batch_size = images1.shape[0]
    psnr_values = []
    
    for i in range(batch_size):
        psnr = calculate_psnr(images1[i], images2[i], max_val)
        psnr_values.append(psnr)
    
    return psnr_values

# ============================================================================
# SSIM (Structural Similarity Index)
# ============================================================================

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, 
                  window_size: int = 11, max_val: float = 1.0) -> float:
    """
    Calculate SSIM between two images.
    
    Args:
        img1, img2: Images as tensors (C, H, W) or (B, C, H, W)
        window_size: Size of sliding window (default 11)
        max_val: Maximum possible pixel value
    
    Returns:
        SSIM value between -1 and 1 (1 is perfect similarity)
    
    Range: -1 to 1 (typically 0.8-0.99 for good quality)
    Use case: Structural and perceptual similarity
    """
    def create_window(window_size: int, channel: int):
        """Create Gaussian window for SSIM calculation"""
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                                for x in range(window_size)])
            return gauss/gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    # Add batch dimension if needed
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    (_, channel, height, width) = img1.size()
    
    # Create window
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    # Calculate means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    # SSIM constants
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()

def batch_ssim(images1: torch.Tensor, images2: torch.Tensor, 
               window_size: int = 11, max_val: float = 1.0) -> List[float]:
    """Calculate SSIM for batch of images"""
    batch_size = images1.shape[0]
    ssim_values = []
    
    for i in range(batch_size):
        ssim = calculate_ssim(images1[i], images2[i], window_size, max_val)
        ssim_values.append(ssim)
    
    return ssim_values

# ============================================================================
# LPIPS (Learned Perceptual Image Patch Similarity)
# ============================================================================

class LPIPSEvaluator:
    """
    LPIPS evaluator using pretrained networks.
    
    Range: 0 to ∞ (typically 0.0-0.8, lower is better)
    Use case: Perceptual similarity using deep features
    """
    
    def __init__(self, net: str = 'alex', device: str = 'cuda'):
        """
        Initialize LPIPS evaluator.
        
        Args:
            net: Network to use ('alex', 'vgg', 'squeeze')
            device: Device to run on
        """
        self.device = device
        self.lpips_fn = lpips.LPIPS(net=net).to(device)
        self.lpips_fn.eval()
    
    def calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Calculate LPIPS between two images.
        
        Args:
            img1, img2: Images as tensors (C, H, W) or (B, C, H, W)
                       Values should be in [-1, 1] range
        
        Returns:
            LPIPS distance (lower is better)
        """
        # Ensure batch dimension
        if len(img1.shape) == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        
        # Move to device
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        # Calculate LPIPS
        with torch.no_grad():
            lpips_distance = self.lpips_fn(img1, img2)
        
        return lpips_distance.mean().item()
    
    def batch_lpips(self, images1: torch.Tensor, images2: torch.Tensor) -> List[float]:
        """Calculate LPIPS for batch of images"""
        batch_size = images1.shape[0]
        lpips_values = []
        
        # Process in smaller batches to save memory
        batch_chunk = 8
        for i in range(0, batch_size, batch_chunk):
            end_idx = min(i + batch_chunk, batch_size)
            chunk1 = images1[i:end_idx]
            chunk2 = images2[i:end_idx]
            
            with torch.no_grad():
                lpips_chunk = self.lpips_fn(chunk1.to(self.device), chunk2.to(self.device))
            
            lpips_values.extend(lpips_chunk.cpu().numpy().flatten())
        
        return lpips_values

# ============================================================================
# FID (Fréchet Inception Distance)
# ============================================================================

class FIDEvaluator:
    """
    FID evaluator for measuring distribution similarity.
    
    Range: 0 to ∞ (typically 1-300, lower is better)
    Use case: Distribution similarity between real and generated images
    """
    
    def __init__(self, device: str = 'cuda', dims: int = 2048):
        """
        Initialize FID evaluator.
        
        Args:
            device: Device to run on
            dims: Dimensionality of Inception features (2048 default)
        """
        self.device = device
        self.dims = dims
        
        # Load Inception model
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.inception_model = InceptionV3([block_idx]).to(device)
        self.inception_model.eval()
    
    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract Inception features from images"""
        features_list = []
        batch_size = 50  # Process in batches to save memory
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(self.device)
                
                # Resize to 299x299 for Inception
                if batch.shape[-1] != 299:
                    batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                
                # Extract features
                features = self.inception_model(batch)[0]
                
                # Remove spatial dimensions
                if features.shape[2] != 1 or features.shape[3] != 1:
                    features = F.adaptive_avg_pool2d(features, output_size=(1, 1))
                
                features = features.cpu().numpy().reshape(features.shape[0], -1)
                features_list.append(features)
        
        return np.concatenate(features_list, axis=0)
    
    def calculate_fid(self, images1: torch.Tensor, images2: torch.Tensor) -> float:
        """
        Calculate FID between two sets of images.
        
        Args:
            images1, images2: Image tensors (B, C, H, W) in range [0, 1]
        
        Returns:
            FID score (lower is better)
        """
        # Extract features
        features1 = self.extract_features(images1)
        features2 = self.extract_features(images2)
        
        # Calculate statistics
        mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
        mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)
        
        # Calculate FID
        fid = self._calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        
        return fid
    
    def _calculate_frechet_distance(self, mu1: np.ndarray, sigma1: np.ndarray, 
                                   mu2: np.ndarray, sigma2: np.ndarray) -> float:
        """Calculate Fréchet distance between two multivariate Gaussians"""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        assert mu1.shape == mu2.shape, f"Mean vectors have different lengths: {mu1.shape} vs {mu2.shape}"
        assert sigma1.shape == sigma2.shape, f"Covariances have different dimensions: {sigma1.shape} vs {sigma2.shape}"
        
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = f"FID calculation produces singular product; adding {1e-6} to diagonal of cov estimates"
            print(msg)
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

# ============================================================================
# COMPREHENSIVE EVALUATION CLASS
# ============================================================================

class ImageQualityEvaluator:
    """Comprehensive image quality evaluator combining all metrics"""
    
    def __init__(self, device: str = 'cuda'):
        """Initialize all evaluators"""
        self.device = device
        self.lpips_evaluator = LPIPSEvaluator(device=device)
        self.fid_evaluator = FIDEvaluator(device=device)
    
    def evaluate_reconstruction(self, original: torch.Tensor, reconstructed: torch.Tensor,
                              max_val: float = 1.0) -> dict:
        """
        Evaluate reconstruction quality using PSNR, SSIM, and LPIPS.
        
        Args:
            original: Original images (B, C, H, W)
            reconstructed: Reconstructed images (B, C, H, W)
            max_val: Maximum pixel value
        
        Returns:
            Dictionary with all metrics
        """
        results = {}
        
        # PSNR
        psnr_values = batch_psnr(original, reconstructed, max_val)
        results['psnr'] = {
            'mean': np.mean(psnr_values),
            'std': np.std(psnr_values),
            'values': psnr_values
        }
        
        # SSIM
        ssim_values = batch_ssim(original, reconstructed, max_val=max_val)
        results['ssim'] = {
            'mean': np.mean(ssim_values),
            'std': np.std(ssim_values),
            'values': ssim_values
        }
        
        # LPIPS (convert to [-1, 1] range if needed)
        if max_val == 1.0:
            orig_lpips = original * 2 - 1
            recon_lpips = reconstructed * 2 - 1
        else:
            orig_lpips = (original / max_val) * 2 - 1
            recon_lpips = (reconstructed / max_val) * 2 - 1
        
        lpips_values = self.lpips_evaluator.batch_lpips(orig_lpips, recon_lpips)
        results['lpips'] = {
            'mean': np.mean(lpips_values),
            'std': np.std(lpips_values),
            'values': lpips_values
        }
        
        return results
    
    def evaluate_generation(self, real_images: torch.Tensor, 
                          generated_images: torch.Tensor) -> dict:
        """
        Evaluate generation quality using FID and LPIPS.
        
        Args:
            real_images: Real images (B, C, H, W) in [0, 1]
            generated_images: Generated images (B, C, H, W) in [0, 1]
        
        Returns:
            Dictionary with FID score
        """
        results = {}
        
        # FID Score
        fid_score = self.fid_evaluator.calculate_fid(real_images, generated_images)
        results['fid'] = fid_score
        
        # Average LPIPS (for diversity measurement)
        # Convert to [-1, 1] range
        real_lpips = real_images * 2 - 1
        gen_lpips = generated_images * 2 - 1
        
        lpips_values = self.lpips_evaluator.batch_lpips(real_lpips, gen_lpips)
        results['lpips_vs_real'] = {
            'mean': np.mean(lpips_values),
            'std': np.std(lpips_values),
            'values': lpips_values
        }
        
        return results

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_images_from_folder(folder_path: str, max_images: int = None) -> torch.Tensor:
    """Load images from folder and convert to tensor"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
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

def print_results(results: dict, title: str = "Results"):
    """Pretty print evaluation results"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print('='*60)
    
    for metric, values in results.items():
        if isinstance(values, dict):
            print(f"{metric.upper():>10}: {values['mean']:.4f} ± {values['std']:.4f}")
        else:
            print(f"{metric.upper():>10}: {values:.4f}")
    
    print('='*60)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_reconstruction_evaluation():
    """Example: Evaluate reconstruction quality"""
    print("Example: Reconstruction Quality Evaluation")
    
    # Create dummy data (replace with your actual images)
    batch_size, channels, height, width = 10, 3, 256, 256
    original = torch.rand(batch_size, channels, height, width)
    
    # Simulate reconstruction with some noise
    noise = torch.randn_like(original) * 0.1
    reconstructed = torch.clamp(original + noise, 0, 1)
    
    # Evaluate
    evaluator = ImageQualityEvaluator()
    results = evaluator.evaluate_reconstruction(original, reconstructed)
    
    print_results(results, "Reconstruction Quality")
    
    return results

def example_generation_evaluation():
    """Example: Evaluate generation quality"""
    print("Example: Generation Quality Evaluation")
    
    # Create dummy data (replace with your actual images)
    batch_size, channels, height, width = 50, 3, 256, 256
    real_images = torch.rand(batch_size, channels, height, width)
    generated_images = torch.rand(batch_size, channels, height, width)
    
    # Evaluate
    evaluator = ImageQualityEvaluator()
    results = evaluator.evaluate_generation(real_images, generated_images)
    
    print_results(results, "Generation Quality")
    
    return results

def example_folder_evaluation():
    """Example: Evaluate images from folders"""
    print("Example: Folder-based Evaluation")
    
    # Example folder paths (replace with your actual paths)
    original_folder = "/path/to/original/images"
    reconstructed_folder = "/path/to/reconstructed/images"
    
    if not (os.path.exists(original_folder) and os.path.exists(reconstructed_folder)):
        print("Note: Example folders don't exist. Using dummy data instead.")
        return example_reconstruction_evaluation()
    
    # Load images
    original_images = load_images_from_folder(original_folder, max_images=100)
    reconstructed_images = load_images_from_folder(reconstructed_folder, max_images=100)
    
    # Evaluate
    evaluator = ImageQualityEvaluator()
    results = evaluator.evaluate_reconstruction(original_images, reconstructed_images)
    
    print_results(results, "Folder-based Evaluation")
    
    return results

if __name__ == "__main__":
    print("Image Quality Metrics Evaluation")
    print("=" * 60)
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run examples
    try:
        print("\n🔄 Running reconstruction evaluation example...")
        recon_results = example_reconstruction_evaluation()
        
        print("\n🎨 Running generation evaluation example...")
        gen_results = example_generation_evaluation()
        
        print("\n📁 Running folder evaluation example...")
        folder_results = example_folder_evaluation()
        
        print("\n✅ All evaluations completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print("Make sure all dependencies are installed:")
        install_requirements()