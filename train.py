import argparse
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import random_split, DataLoader
import torchvision
from torchvision.models import resnet18
from torchvision.transforms import v2
from tqdm import tqdm

from netvlad import NetVLAD
from netvlad import EmbedNet
from hard_triplet_loss import HardTripletLoss
from utils import CCUDataset, TransformWrapper

parser = argparse.ArgumentParser(description="NetVLAD training args")
parser.add_argument("--bs", default=16, type=int, help="batch size")
parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
parser.add_argument("--gpu", action="store_true", help="use gpu")
parser.add_argument(
    "--data_dir", default="./data", type=str, help="training dataset dir path"
)
parser.add_argument(
    "--sat_path", default="./data/sat.jpeg", type=str, help="the satellite image path"
)
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--stride", default=2, type=int, help="stride")
parser.add_argument("--pad", default=10, type=int, help="padding")
parser.add_argument("--h", default=224, type=int, help="height")
parser.add_argument("--w", default=224, type=int, help="width")
parser.add_argument(
    "--resume",
    default=None,
    type=str,
    help="resume traing model, if not none then use the model to keep traing",
)
parser.add_argument(
    "--output", default="./models/", type=str, help="Where to save the trained model"
)
parser.add_argument("--name", type=str, help="model name")
parser.add_argument(
    "--pool",
    default="avg",
    type=str,
    help="pooling mode, default: avg, options: avg, max",
)
parser.add_argument("--droprate", default=0.5, type=float, help="droprate")
parser.add_argument("--ca", action="store_true", help="use Color data Augmentation")
parser.add_argument("--blocks", default=6, type=int, help="the number of blocks")
opts = parser.parse_args()

transforms = v2.Compose(
    [
        v2.Resize(
            (opts.h, opts.w),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        ),
        v2.Pad(opts.pad, padding_mode="edge"),
        v2.RandomCrop((opts.h, opts.w)),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1, hue=0),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

valid_transforms = v2.Compose(
    [
        v2.Resize(
            (opts.h, opts.w),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        ),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

full_dataset = CCUDataset(opts.data_dir, transforms)
# Set the split ratio
train_ratio = 0.8
val_ratio = 0.2

# Calculate the number of samples for each split
num_samples = len(full_dataset)
num_train = int(num_samples * train_ratio)
num_val = num_samples - num_train

train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])

val_dataset = TransformWrapper(val_dataset, valid_transforms)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=opts.bs,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)
valid_dataloader = DataLoader(
    val_dataset, batch_size=opts.bs, shuffle=False, num_workers=8
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Discard layers at the end of base network
encoder = resnet18(pretrained=True)
base_model = nn.Sequential(
        encoder.conv1,
        encoder.bn1,
        encoder.relu,
        encoder.maxpool,
        encoder.layer1,
        encoder.layer2,
        encoder.layer3,
        encoder.layer4,
)
dim = list(base_model.parameters())[-1].shape[0]  # last channels (512)

# Define model for embedding
net_vlad = NetVLAD(num_clusters=32, dim=dim, alpha=1.0)
model = EmbedNet(base_model, net_vlad).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, momentum=0.9)
# Define loss
criterion = HardTripletLoss(margin=0.1).to(device)

pbar = tqdm(range(opts.start_epoch, opts.epochs))
for epoch in pbar:
    train_loss = 0.0
    model.train()

    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        print(f"inputs shape: {inputs.shape}, labels shape: {labels.shape}, outputs shape: {outputs.shape}\n")
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for inputs, labels in valid_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, label)
            valid_loss += loss.item()


# This is just toy example. Typically, the number of samples in each classes are 4.
labels = torch.randint(0, 10, (40,)).long()
x = torch.rand(40, 3, 128, 128).to(device)
output = model(x)

triplet_loss = criterion(output, labels)
