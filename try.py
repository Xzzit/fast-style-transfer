import os
import time
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import utils
from transformer_net import TransformerNet
from vgg import Vgg16


device = torch.device("cuda")
dataset = 'D:/Project/PyPro/data/coco2017/val2017'
style_image = 'D:/Project/PyPro/data/inkwash.jpg'
epochs = 1
image_size = 256
batch_size = 4


vgg = Vgg16(requires_grad=False).to(device)
style_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])
style = utils.load_image(style_image, size=image_size)
style = style_transform(style)
style = style.repeat(1, 1, 1, 1).to(device)
features_style = vgg(utils.normalize_batch(style))
gram_style = [utils.gram_matrix(y) for y in features_style]

transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


train_dataset = CustomDataSet(dataset, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size)

transformer = TransformerNet().to(device)
optimizer = Adam(transformer.parameters(), 1e-3)
mse_loss = torch.nn.MSELoss()

for e in range(epochs):
    # transformer.train()
    # agg_content_loss = 0.
    # agg_style_loss = 0.
    # count = 0
    for batch_id, x in enumerate(train_loader):
        # count += batch_size
        # optimizer.zero_grad()

        x = x.to(device)
        y = transformer(x)

        y = utils.normalize_batch(y)
        # x = utils.normalize_batch(x)

        features_y = vgg(y)
        # features_x = vgg(x)
        #
        # content_loss = 1e10 * mse_loss(features_y.relu2_2, features_x.relu2_2)

        style_loss = 0.
        for ft_y, gm_s in zip(features_y, gram_style):
            gm_y = utils.gram_matrix(ft_y)
            gm_s = gm_s.repeat(ft_y.shape[0], 1, 1)
            style_loss += mse_loss(gm_y, gm_s)
            print(gm_y.shape)
            print(gm_s.shape)
        # style_loss *= 1e5

        # noise = torch.randn(batch_size, 3, image_size, image_size) - 1/2
        # noise_x = x + noise.to(device)
        # features_noise_y = vgg(noise_x)
        #
        # pop_loss = 0
        # for y, ny in zip(features_y, features_noise_y):
        #     pop_loss += mse_loss(y, ny)
        # pop_loss *= 1e3
        #
        # total_loss = content_loss + style_loss + pop_loss
        # total_loss.backward()
        # optimizer.step()
        #
        # agg_content_loss += content_loss.item()
        # agg_style_loss += style_loss.item()
