import os
import time
import argparse
from PIL import Image

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.onnx

import utils
from models.autoencoder import Autoencoder
from models.bottleNet import BottleNetwork
from vgg import Vgg16


# Define custom dataset
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


def train(dataset, style_image, save_model_dir, epochs,
          content_weight=1e5, style_weight=1e10, pop_weight=1e1, image_size=256, batch_size=16,
          model_name='inkwash'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(42)
    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda a: a.mul(255))
    ])
    train_dataset = CustomDataSet(dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    transformer = BottleNetwork().to(device)
    optimizer = Adam(transformer.parameters(), 1e-3)
    mse_loss = torch.nn.MSELoss()

    # Define pre-trained VGG net & style image's output in VGG
    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(style_image, size=[image_size, image_size])
    style = style_transform(style)
    style = style.repeat(1, 1, 1, 1).to(device)
    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    # Train transformer net to match the style
    for e in range(epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        agg_pop_loss = 0.
        count = 0
        for batch_id, x in enumerate(train_loader):
            count += batch_size
            optimizer.zero_grad()

            '''
            transformer is the network which we use to stylize images
            x: original image (In this case, images in COCO dataset)
            y: stylized image
            noise_x: original image with noise
            noise_y: stylized noise_x
            '''
            x = x.to(device)
            y = transformer(x)

            # Add noise to original image to increase the consistency
            noise = 1 / 6 * torch.randn(x.shape[0], 3, image_size, image_size)
            noise_x = x + noise.to(device)

            x = utils.normalize_batch(x)
            y = utils.normalize_batch(y)

            # Get the representation of origin and stylized images in VGG
            features_y = vgg(y)
            features_x = vgg(x)

            # Compute content loss
            content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            # Compute style loss
            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                gm_s = gm_s.repeat(ft_y.shape[0], 1, 1)
                style_loss += mse_loss(gm_y, gm_s)
            style_loss *= style_weight

            # Compute the noise item loss
            noise_y = transformer(noise_x)
            pop_loss = pop_weight * mse_loss(noise_y, y)

            # Compute the total loss
            total_loss = content_loss + style_loss + pop_loss
            total_loss.backward()
            optimizer.step()

            # Output the loss information
            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            agg_pop_loss += pop_loss.item()

            if (batch_id + 1) % 500 == 0:
                message = "{}\tEpoch {}: [{}/{}]\tcontent: {:.3f}\tstyle: {:.3f}\tpop: {:.3f}\ttotal: {:.3f}".format(
                    time.ctime(),
                    e + 1,
                    count,
                    len(train_dataset),
                    agg_content_loss / (batch_id + 1),
                    agg_style_loss / (batch_id + 1),
                    agg_pop_loss / (batch_id + 1),
                    (agg_content_loss + agg_style_loss + agg_pop_loss) / (batch_id + 1)
                )
                print(message)

        # Save model
        transformer.eval()
        save_model_filename = model_name + '_' + \
                              'c' + format(content_weight, '.0E').replace('+', '') + '_' + \
                              's' + format(style_weight, '.0E').replace('+', '') + '_' + \
                              'p' + format(pop_weight, '.0E').replace('+', '') + \
                              ".pth"
        save_model_path = os.path.join(save_model_dir, save_model_filename)
        torch.save(transformer.state_dict(), save_model_path)
        print("\nTrained model saved at ", save_model_path)


def main():

    # Arguments for training
    train_arg_parser = argparse.ArgumentParser(description="parser for fast neural style transfer")
    train_arg_parser.add_argument("--epochs", type=int, default=1,
                                  help="number of training epochs, default is 1")
    train_arg_parser.add_argument("--batch-size", type=int, default=16,
                                  help="batch size for training, default is 16")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, required=True,
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 x 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--consistency-weight", type=float, default=1e1,
                                  help="weight for consistency-loss, default is 1e1")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")


if __name__ == "__main__":
    main()
