import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm
from PIL import Image


def save_checkpoint(model, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    torch.save(checkpoint, path)


def create_dataset(satellite_img_path, output_dir=None, nums=10000):
    imgs = []
    sat_img = Image.open(satellite_img_path)
    original_width, original_height = sat_img.size
    new_width = original_width * 4
    new_height = original_height * 4
    transforms = v2.Compose([
        v2.Resize((new_height, new_width)),
        v2.RandomCrop(size=(1024, 1024)),
    ])

    for num in tqdm(range(nums), desc="random crop satellite image"):
        map = sat_img.copy()
        img = transforms(map)
        img.save(f"{output_dir}/{num:04d}.jpg")
        imgs.append(img)
    return imgs

def load_imgs(sat_img_folder):
    files = os.listdir(sat_img_folder)
    imgs = [f"{sat_img_folder}/{f}" for f in files if os.path.isfile(f"{sat_img_folder}/{f}") and f.endswith('.jpg')]
    return imgs


class CCUDataset(Dataset):
    def __init__(self, sat_img_folder, transform):
        self.imgs = load_imgs(sat_img_folder)
        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.imgs[idx]
        image = Image.open(image_path)
        label = Image.open(image_path)
        # print(f"image path: {image_path}, shape: {image.size[0]}, {image.size[1]}")
        img = self.transform(image)
        label = self.transform(label)
        # print(f"img shape: {img.shape}, label shape: {label.shape}")

        return img, label

    def __len__(self):
        return len(self.imgs)


class TransformWrapper(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.transform(item)
