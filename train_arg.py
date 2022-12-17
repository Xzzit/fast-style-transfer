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


def train(args):

    # Set device
    if args.mps:
        device = torch.device('mps')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Define dataloader transform function
    content_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda a: a.mul(255))
    ])

    # Initialize dataloader
    train_dataset = CustomDataSet(args.dataset, content_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    # Initialize fast neural style transfer model and optimizer
    transformer = BottleNetwork().to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    # Define pre-trained VGG net
    vgg = Vgg16(requires_grad=False).to(device)

    # Define transform function of style image loader
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    # Load style image
    style = utils.load_image(args.style_image, size=[args.image_size, args.image_size])
    style = style_transform(style)
    style = style.repeat(1, 1, 1, 1).to(device)

    # Compute style features
    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    # Train style transfer net to match the style
    for e in range(args.epochs):

        # Initialize model and loss
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        agg_consistency_loss = 0.
        count = 0

        for batch_id, x in enumerate(train_loader):
            '''
            transformer is the network which we use to stylize images
            x: original image (In this case, images in COCO dataset)
            y: stylized image
            noise_x: original image with Gaussian noise
            noise_y: stylized noise_x. aka transformer(noise_x)
            '''

            count += args.batch_size
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            # Add noise to original image to increase the consistency
            noise = 1 / 6 * torch.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
            noise_x = x + noise.to(device)

            # Normalize batches
            x = utils.normalize_batch(x)
            y = utils.normalize_batch(y)

            # Get the content representation of origin and stylized images in VGG
            features_y = vgg(y)
            features_x = vgg(x)

            # Compute content loss
            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            # Compute style loss
            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                gm_s = gm_s.repeat(ft_y.shape[0], 1, 1)
                style_loss += mse_loss(gm_y, gm_s)
            style_loss *= args.style_weight

            # Compute the noise item loss
            noise_y = transformer(noise_x)
            consistency_loss = args.consistency_weight * mse_loss(noise_y, y)

            # Compute the total loss
            total_loss = content_loss + style_loss + consistency_loss
            total_loss.backward()
            optimizer.step()

            # Output the loss information
            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            agg_consistency_loss += consistency_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                message = "{}\tEpoch {}: [{}/{}]\tcontent: {:.3f}\tstyle: {:.3f}\tconsistency: {:.3f}\ttotal: {:.3f}".format(
                    time.ctime(),
                    e + 1,
                    count,
                    len(train_dataset),
                    agg_content_loss / (batch_id + 1),
                    agg_style_loss / (batch_id + 1),
                    agg_consistency_loss / (batch_id + 1),
                    (agg_content_loss + agg_style_loss + agg_consistency_loss) / (batch_id + 1)
                )
                print(message)

        # Save model
        transformer.eval()
        save_model_filename = args.model_name + '_' + \
                              'cont' + format(args.content_weight, '.0E').replace('+', '') + '_' + \
                              'sty' + format(args.style_weight, '.0E').replace('+', '') + '_' + \
                              'cons' + format(args.consistency_weight, '.0E').replace('+', '') + \
                              ".pth"
        save_model_path = os.path.join(args.save_model_dir, save_model_filename)
        torch.save(transformer.state_dict(), save_model_path)
        print("\nTrained model saved at ", save_model_path)


def main():

    # Arguments for training
    train_arg_parser = argparse.ArgumentParser(description="parser for fast neural style transfer")

    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, required=True,
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--model-name", type=str, default='Name',
                                  help="name for saved model.")

    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--consistency-weight", type=float, default=1e1,
                                  help="weight for consistency-loss, default is 1e1")

    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 x 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--batch-size", type=int, default=16,
                                  help="batch size for training, default is 16")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--epochs", type=int, default=1,
                                  help="number of training epochs, default is 1")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")

    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")

    train_arg_parser.add_argument('--mps', action='store_true', default=False, help='enable macOS GPU training')

    args = train_arg_parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
