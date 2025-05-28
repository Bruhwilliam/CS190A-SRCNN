import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# Dataset Class
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
        return lr, hr

# SRCNN Model
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


# Visualization
def show_image(tensor, title="Image"):
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    plt.imshow(np.clip(img, 0, 1))
    plt.title(title)
    plt.axis('off')
    plt.show()

def save_example_output(model, dataset, device, output_path="comparison.png"):
    model.eval()
    with torch.no_grad():
        lr, hr = dataset[0]
        lr_input = lr.unsqueeze(0).to(device)
        sr = model(lr_input)
        sr = sr.squeeze(0).cpu()
        images = torch.stack([lr, sr, hr], dim=0)
        save_image(images, output_path, nrow=3)
        print(f"✅ Saved: {output_path}")


# Training with Resume Support
def train_model():
    hr_dir = "Images"
    lr_dir = "Processed/lr_images"
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset = SRDataset(hr_dir, lr_dir, transform=transform)
    dataset.image_pairs = dataset.image_pairs[:300]  # adjust for your dataset
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    start_epoch = 0
    if os.path.exists("srcnn_model.pth"):
        print("📂 Found existing model weights. Loading...")
        model.load_state_dict(torch.load("srcnn_model.pth"))
    else:
        print("🆕 No saved model found. Starting from scratch.")

    if os.path.exists("last_epoch.txt"):
        with open("last_epoch.txt", "r") as f:
            start_epoch = int(f.read().strip()) + 1

    total_epochs = 100
    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_loss = 0
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = criterion(sr, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"📈 Epoch {epoch+1}/{total_epochs} - Loss: {epoch_loss / len(loader):.4f}")

        torch.save(model.state_dict(), "srcnn_model.pth")
        with open("last_epoch.txt", "w") as f:
            f.write(str(epoch))

        if (epoch + 1) % 10 == 0:
            save_example_output(model, dataset, device, f"comparison_epoch_{epoch+1}.png")

    return model, dataset, device


# visualization for manual view
def visualize_sample(model, dataset, device):
    model.eval()
    with torch.no_grad():
        lr, hr = dataset[0]
        sr = model(lr.unsqueeze(0).to(device))[0]
        show_image(lr, "Low-Resolution Input")
        show_image(sr, "Super-Resolved Output")
        show_image(hr, "High-Resolution Ground Truth")


def run_on_any_image_with_hr(lr_path, hr_path, model_path="srcnn_model.pth", output_path="result.png"):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Preprocess input images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    lr_img = Image.open(lr_path).convert("RGB")
    hr_img = Image.open(hr_path).convert("RGB")

    lr = transform(lr_img).unsqueeze(0).to(device)
    hr = transform(hr_img)  # just keep on CPU

    # Super-resolve
    with torch.no_grad():
        sr = model(lr)[0].cpu()

    # Save side-by-side comparison
    save_image(torch.stack([lr[0].cpu(), sr, hr]), output_path, nrow=3)
    print(f"✅ Saved side-by-side result (LR | SR | HR): {output_path}")

# Main
if __name__ == "__main__":
    run_on_any_image_with_hr(
        lr_path="Processed/lr_images/tenniscourt/tenniscourt03.tif",
        hr_path="Images/tenniscourt/tenniscourt03.tif",
        output_path="side_by_side3.png"
    )
    # model, dataset, device = train_model()
    # visualize_sample(model, dataset, device)