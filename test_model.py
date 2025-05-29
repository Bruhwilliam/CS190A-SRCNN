import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tqdm import tqdm
import time
from datetime import datetime

# Import the SRCNN model class
from train import SRCNN

def plot_metrics(per_image_metrics, output_dir):
    """Create and save plots of PSNR and SSIM metrics with enhanced visualizations"""
    # Extract data
    images = [m['image'] for m in per_image_metrics]
    psnr_values = [m['psnr'] for m in per_image_metrics]
    ssim_values = [m['ssim'] for m in per_image_metrics]
    
    # Create figure with subplots
    plt.style.use('bmh')  # Use a built-in style instead of seaborn
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Bar plot of PSNR and SSIM for each image (with error bars)
    ax1 = plt.subplot(331)
    x = np.arange(len(images))
    width = 0.35
    
    # Calculate standard deviations if we have enough samples
    psnr_std = np.std(psnr_values) if len(psnr_values) > 1 else 0
    ssim_std = np.std(ssim_values) if len(ssim_values) > 1 else 0
    
    ax1.bar(x - width/2, psnr_values, width, label='PSNR (dB)', color='skyblue', 
            yerr=psnr_std, capsize=5)
    ax1.bar(x + width/2, ssim_values, width, label='SSIM', color='lightgreen',
            yerr=ssim_std, capsize=5)
    
    ax1.set_xlabel('Images')
    ax1.set_ylabel('Value')
    ax1.set_title('PSNR and SSIM for Each Image\nwith Standard Deviation')
    ax1.set_xticks(x)
    ax1.set_xticklabels(images, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Running average plot with confidence intervals
    ax2 = plt.subplot(332)
    running_avg_psnr = np.cumsum(psnr_values) / np.arange(1, len(psnr_values) + 1)
    running_avg_ssim = np.cumsum(ssim_values) / np.arange(1, len(ssim_values) + 1)
    
    # Calculate running standard deviations
    running_std_psnr = [np.std(psnr_values[:i+1]) for i in range(len(psnr_values))]
    running_std_ssim = [np.std(ssim_values[:i+1]) for i in range(len(ssim_values))]
    
    ax2.plot(range(1, len(images) + 1), running_avg_psnr, 'b-', label='Running Avg PSNR')
    ax2.plot(range(1, len(images) + 1), running_avg_ssim, 'g-', label='Running Avg SSIM')
    ax2.fill_between(range(1, len(images) + 1), 
                     np.array(running_avg_psnr) - np.array(running_std_psnr),
                     np.array(running_avg_psnr) + np.array(running_std_psnr),
                     alpha=0.2, color='blue')
    ax2.fill_between(range(1, len(images) + 1),
                     np.array(running_avg_ssim) - np.array(running_std_ssim),
                     np.array(running_avg_ssim) + np.array(running_std_ssim),
                     alpha=0.2, color='green')
    
    ax2.set_xlabel('Number of Images')
    ax2.set_ylabel('Value')
    ax2.set_title('Running Average with Confidence Intervals')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Enhanced scatter plot with regression line
    ax3 = plt.subplot(333)
    scatter = ax3.scatter(psnr_values, ssim_values, c=range(len(images)), 
                         cmap='viridis', s=100, alpha=0.6)
    
    # Add regression line
    z = np.polyfit(psnr_values, ssim_values, 1)
    p = np.poly1d(z)
    ax3.plot(psnr_values, p(psnr_values), "r--", alpha=0.8)
    
    # Add correlation coefficient
    corr = np.corrcoef(psnr_values, ssim_values)[0,1]
    ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
             transform=ax3.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    ax3.set_xlabel('PSNR (dB)')
    ax3.set_ylabel('SSIM')
    ax3.set_title('PSNR vs SSIM with Regression Line')
    plt.colorbar(scatter, ax=ax3, label='Image Index')
    ax3.grid(True, alpha=0.3)
    
    # 4. Enhanced box plot with individual points
    ax4 = plt.subplot(334)
    box = ax4.boxplot([psnr_values, ssim_values], labels=['PSNR', 'SSIM'],
                      patch_artist=True)
    
    # Add individual points
    for i, data in enumerate([psnr_values, ssim_values]):
        ax4.scatter([i+1]*len(data), data, alpha=0.4, color='red')
    
    # Customize box colors
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    ax4.set_title('Distribution of PSNR and SSIM\nwith Individual Points')
    ax4.grid(True, alpha=0.3)
    
    # 5. Histogram of PSNR values
    ax5 = plt.subplot(335)
    ax5.hist(psnr_values, bins=10, color='skyblue', alpha=0.7)
    ax5.axvline(np.mean(psnr_values), color='red', linestyle='dashed', 
                label=f'Mean: {np.mean(psnr_values):.2f}')
    ax5.set_xlabel('PSNR (dB)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Histogram of PSNR Values')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Histogram of SSIM values
    ax6 = plt.subplot(336)
    ax6.hist(ssim_values, bins=10, color='lightgreen', alpha=0.7)
    ax6.axvline(np.mean(ssim_values), color='red', linestyle='dashed',
                label=f'Mean: {np.mean(ssim_values):.4f}')
    ax6.set_xlabel('SSIM')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Histogram of SSIM Values')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Cumulative Distribution Function (CDF)
    ax7 = plt.subplot(337)
    for metric, color, label in zip([psnr_values, ssim_values], 
                                  ['skyblue', 'lightgreen'],
                                  ['PSNR', 'SSIM']):
        sorted_data = np.sort(metric)
        p = 1. * np.arange(len(metric)) / (len(metric) - 1)
        ax7.plot(sorted_data, p, color=color, label=label)
    
    ax7.set_xlabel('Value')
    ax7.set_ylabel('Cumulative Probability')
    ax7.set_title('Cumulative Distribution Function')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Violin plot
    ax8 = plt.subplot(338)
    violin = ax8.violinplot([psnr_values, ssim_values], showmeans=True)
    ax8.set_xticks([1, 2])
    ax8.set_xticklabels(['PSNR', 'SSIM'])
    ax8.set_title('Violin Plot of PSNR and SSIM')
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary statistics
    ax9 = plt.subplot(339)
    ax9.axis('off')
    stats_text = (
        f"Summary Statistics:\n\n"
        f"PSNR:\n"
        f"  Mean: {np.mean(psnr_values):.2f} dB\n"
        f"  Std: {np.std(psnr_values):.2f} dB\n"
        f"  Min: {np.min(psnr_values):.2f} dB\n"
        f"  Max: {np.max(psnr_values):.2f} dB\n\n"
        f"SSIM:\n"
        f"  Mean: {np.mean(ssim_values):.4f}\n"
        f"  Std: {np.std(ssim_values):.4f}\n"
        f"  Min: {np.min(ssim_values):.4f}\n"
        f"  Max: {np.max(ssim_values):.4f}\n\n"
        f"Correlation: {corr:.3f}"
    )
    ax9.text(0.1, 0.5, stats_text, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_metrics(sr_image, hr_image):
    """Calculate PSNR and SSIM metrics"""
    # Convert to numpy arrays
    sr_np = sr_image.detach().cpu().numpy()
    hr_np = hr_image.detach().cpu().numpy()
    
    # Calculate MSE
    mse = np.mean((sr_np - hr_np) ** 2)
    if mse == 0:
        return float('inf'), 1.0
    
    # Calculate PSNR
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    
    # Calculate SSIM (simplified version)
    c1 = (0.01 * max_pixel) ** 2
    c2 = (0.03 * max_pixel) ** 2
    
    mu_sr = np.mean(sr_np)
    mu_hr = np.mean(hr_np)
    sigma_sr = np.var(sr_np)
    sigma_hr = np.var(hr_np)
    sigma_srhr = np.mean((sr_np - mu_sr) * (hr_np - mu_hr))
    
    ssim = ((2 * mu_sr * mu_hr + c1) * (2 * sigma_srhr + c2)) / \
           ((mu_sr ** 2 + mu_hr ** 2 + c1) * (sigma_sr + sigma_hr + c2))
    
    return psnr, ssim

def show_image(tensor, title="Image", subplot=None):
    """Display a tensor as an image in the specified subplot"""
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    if subplot:
        plt.subplot(subplot)
    plt.imshow(np.clip(img, 0, 1))
    plt.title(title)
    plt.axis('off')

def load_model(model_path="srcnn_model.pth", device="cpu"):
    """Load the trained SRCNN model"""
    try:
        model = SRCNN()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def process_image(image_path, model, device="cpu"):
    """Process a single image through the model"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        image_tensor = transform(image).unsqueeze(0)
        
        # Move to device and process
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            sr_image = model(image_tensor)[0]
        
        return image_tensor[0], sr_image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

def save_results(lr_image, sr_image, hr_image, metrics, output_dir, image_name):
    """Save the results and metrics to files"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the images
    plt.figure(figsize=(15, 5))
    show_image(lr_image, "Low-Resolution Input", 131)
    show_image(sr_image, "Super-Resolved Output", 132)
    show_image(hr_image, "High-Resolution Ground Truth", 133)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f"{image_name}_comparison.png"))
    plt.close()
    
    # Save metrics
    with open(os.path.join(output_dir, f"{image_name}_metrics.txt"), "w") as f:
        f.write(f"PSNR: {metrics['psnr']:.2f} dB\n")
        f.write(f"SSIM: {metrics['ssim']:.4f}\n")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test SRCNN model')
    parser.add_argument('--model', type=str, default="srcnn_model.pth",
                      help='Path to the trained model')
    parser.add_argument('--num_images', type=int, default=100,
                      help='Number of test images to process')
    parser.add_argument('--output_dir', type=str, default="test_results",
                      help='Directory to save results')
    parser.add_argument('--gpu', action='store_true',
                      help='Use GPU if available')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.model, device)
    if model is None:
        return
    
    # Get a list of test images
    test_dir = "Images"
    test_images = []
    
    # Walk through the Images directory and collect test images
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith(".tif"):
                test_images.append(os.path.join(root, file))
                if len(test_images) >= args.num_images:
                    break
        if len(test_images) >= args.num_images:
            break
    
    if len(test_images) < args.num_images:
        print(f"Warning: Only found {len(test_images)} images, which is less than the requested {args.num_images}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each test image
    total_psnr = 0
    total_ssim = 0
    per_image_metrics = []
    
    print(f"\nProcessing {len(test_images)} images...")
    for idx, image_path in enumerate(tqdm(test_images, desc="Processing images")):
        # Get the corresponding low-res image
        lr_path = image_path.replace("Images", "Processed/lr_images")
        
        # Process both images
        lr_image, sr_image = process_image(lr_path, model, device)
        if lr_image is None or sr_image is None:
            continue
            
        hr_image = Image.open(image_path).convert("RGB")
        hr_image = transforms.Resize((256, 256))(hr_image)
        hr_image = transforms.ToTensor()(hr_image)
        
        # Calculate metrics
        psnr, ssim = calculate_metrics(sr_image, hr_image)
        metrics = {'psnr': psnr, 'ssim': ssim}
        total_psnr += psnr
        total_ssim += ssim
        per_image_metrics.append({
            'image': os.path.basename(image_path),
            'psnr': psnr,
            'ssim': ssim
        })
        
        # Save results
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        save_results(lr_image, sr_image, hr_image, metrics, output_dir, image_name)
        
        # Only display the first 5 images as examples
        if idx < 5:
            plt.figure(figsize=(15, 5))
            plt.suptitle(f"Example {idx + 1}: {os.path.basename(image_path)} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}", 
                        fontsize=14)
            show_image(lr_image, f"Low-Resolution Input", 131)
            show_image(sr_image, "Super-Resolved Output", 132)
            show_image(hr_image, "High-Resolution Ground Truth", 133)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
    
    # Print per-image metrics
    if per_image_metrics:
        print("\nPer-image PSNR and SSIM:")
        print(f"{'Image':30s} {'PSNR (dB)':>10s} {'SSIM':>10s}")
        print("-" * 54)
        for m in per_image_metrics:
            print(f"{m['image']:30s} {m['psnr']:10.2f} {m['ssim']:10.4f}")
    
    # Print average metrics
    num_images = len(per_image_metrics)
    if num_images > 0:
        avg_psnr = total_psnr / num_images
        avg_ssim = total_ssim / num_images
        print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        
        # Save average metrics
        with open(os.path.join(output_dir, "average_metrics.txt"), "w") as f:
            f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        
        # Create and save metric plots
        plot_metrics(per_image_metrics, output_dir)
        print(f"\nSaved metric plots to {output_dir}/metrics_analysis.png")

if __name__ == "__main__":
    main() 