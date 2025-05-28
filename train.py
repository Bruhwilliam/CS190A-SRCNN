
import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# ------------------------------
# Dataset Class
# ------------------------------
class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.transform = transform
        self.image_pairs = []

        for class_name in os.listdir(hr_dir):
            hr_class_path = os.path.join(hr_dir, class_name)
            lr_class_path = os.path.join(lr_dir, class_name)
            if not os.path.isdir(hr_class_path):
                continue
            for img_name in os.listdir(hr_class_path):
                self.image_pairs.append((
                    os.path.join(lr_class_path, img_name),
                    os.path.join(hr_class_path, img_name)
                ))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        lr_path, hr_path = self.image_pairs[idx]
        lr = Image.open(lr_path).convert("RGB")
        hr = Image.open(hr_path).convert("RGB")

        if self.transform:
            lr = self.transform(lr)
            hr = self.transform(hr)
        print(f"Loaded pair: {lr.shape}, {hr.shape}")
        return lr, hr

# ------------------------------
# SRCNN Model
# ------------------------------
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.layers(x)

# ------------------------------
# Visualization Helper
# ------------------------------
def show_image(tensor, title="Image"):
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    plt.imshow(np.clip(img, 0, 1))
    plt.title(title)
    plt.axis('off')
    plt.show()

# ------------------------------
# Training Script
# ------------------------------
def train_model():
    hr_dir = "Images"
    lr_dir = "Processed/lr_images"
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset = SRDataset(hr_dir, lr_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        epoch_loss = 0
        for batch_idx, (lr, hr) in enumerate(loader):
            print(f"Processing batch {batch_idx + 1}...")
            lr, hr = lr.to(device), hr.to(device)

            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/10], Loss: {epoch_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "srcnn_model.pth")
    return model, dataset, device

# ------------------------------
# Inference & Visualization
# ------------------------------
def visualize_sample(model, dataset, device):
    model.eval()
    with torch.no_grad():
        lr, hr = dataset[0]
        sr = model(lr.unsqueeze(0).to(device))[0]
        show_image(lr, "Low-Resolution Input")
        show_image(sr, "Super-Resolved Output")
        show_image(hr, "High-Resolution Ground Truth")

if __name__ == "__main__":
    model, dataset, device = train_model()
    visualize_sample(model, dataset, device)
