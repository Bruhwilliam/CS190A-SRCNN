import os
from PIL import Image

HR_ROOT = "Images"
LR_ROOT = "Processed/lr_images"
SCALE = 2  # Downsampling factor

def make_dirs():
    for class_dir in os.listdir(HR_ROOT):
        hr_path = os.path.join(HR_ROOT, class_dir)
        lr_path = os.path.join(LR_ROOT, class_dir)
        if os.path.isdir(hr_path):
            os.makedirs(lr_path, exist_ok=True)

def make_lr_image(img, scale=2):
    w, h = img.size
    lr_down = img.resize((w // scale, h // scale), Image.BICUBIC)
    lr_up = lr_down.resize((w, h), Image.BICUBIC)
    return lr_up

def process_images():
    for class_dir in os.listdir(HR_ROOT):
        hr_class_path = os.path.join(HR_ROOT, class_dir)
        lr_class_path = os.path.join(LR_ROOT, class_dir)
        if not os.path.isdir(hr_class_path):
            continue
        for filename in os.listdir(hr_class_path):
            if filename.endswith(".tif"):
                hr_path = os.path.join(hr_class_path, filename)
                lr_path = os.path.join(lr_class_path, filename)
                img = Image.open(hr_path).convert('RGB')
                lr_img = make_lr_image(img, SCALE)
                lr_img.save(lr_path)
                print(f"Processed {filename} → {lr_path}")

if __name__ == "__main__":
    make_dirs()
    process_images()