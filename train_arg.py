import os
import sys
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
from models import *
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

    # Initialize fast neural style transfer model
    if args.model_type == 'ae':
        stylize_network = Autoencoder().to(device)
    elif args.model_type == 'bo':
        stylize_network = BottleNetwork().to(device)
    elif args.model_type == 'res':
        stylize_network = ResNext().to(device)
    elif args.model_type == 'dense':
        stylize_network = DenseNet().to(device)
    else:
        print('Error: invalid selected architecture')
        sys.exit()

    # Initialize optimizer
    optimizer = Adam(stylize_network.parameters(), args.lr)
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
    gram_style = [utils.gram_matrix(f) for f in features_style]

    # Train style transfer net to match the style
    for e in range(args.epochs):

        # Initialize model and loss
        stylize_network.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        agg_consistency_loss = 0.
        agg_tv_loss = 0.
        count = 0

        for batch_id, content_img in enumerate(train_loader):
            '''
            stylize_network is the network which we use to stylize images
            content_img: original image (In this case, images in COCO dataset)
            generated_img: stylized image
            noise_x: original image with Gaussian noise
            noise_y: stylized noise_x. aka stylize_network(noise_x)
            '''

            count += args.batch_size
            optimizer.zero_grad()

            content_img = content_img.to(device)
            generated_img = stylize_network(content_img)

            # Add noise to original image to increase the consistency
            content_img_noise = 0.95 * content_img + 255 * 0.05 * torch.randn_like(content_img)

            # Get the content representation of origin and stylized images in 
            features_content = vgg(utils.normalize_batch(content_img))
            features_generated = vgg(utils.normalize_batch(generated_img))

            # Compute content loss
            content_loss = args.content_weight * mse_loss(features_generated.relu2_2, features_content.relu2_2)

            # Compute style loss
            style_loss = 0.
            for ft_y, gm_s in zip(features_generated, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                gm_s = gm_s.repeat(ft_y.shape[0], 1, 1)
                style_loss += mse_loss(gm_y, gm_s)
            style_loss *= args.style_weight

            # Compute the noise item loss
            generated_noise = stylize_network(content_img_noise)
            consistency_loss = args.consistency_weight * mse_loss(generated_noise, generated_img)

            # Compute the tv loss
            tv_loss = args.tv_weight * utils.tv_loss(generated_img)

            # Compute the total loss
            total_loss = content_loss + style_loss + consistency_loss
            total_loss.backward()
            optimizer.step()

            # Output the loss information
            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            agg_consistency_loss += consistency_loss.item()
            agg_tv_loss += tv_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                message = "{}\tEpoch {}: [{}/{}]\tcontent: {:.3f}\tstyle: {:.3f}\tconsistency: {:.3f}\ttv loss: {:.3f}\ttotal: {:.3f}".format(
                    time.ctime(),
                    e + 1,
                    count,
                    len(train_dataset),
                    agg_content_loss / (batch_id + 1),
                    agg_style_loss / (batch_id + 1),
                    agg_consistency_loss / (batch_id + 1),
                    agg_tv_loss / (batch_id + 1),
                    (agg_content_loss + agg_style_loss + agg_consistency_loss + agg_tv_loss) / (batch_id + 1)
                )
                print(message)

        # Save model
        stylize_network.eval()
        if args.model_name is None:
            args.model_name = os.path.splitext(os.path.basename(args.style_image))[0]
        
        save_model_filename = args.model_name + '_' + \
                            'cont'    + format(args.content_weight, '.1E').replace('+', '').replace('.', 'p') + '_' + \
                            'sty'     + format(args.style_weight, '.1E').replace('+', '').replace('.', 'p')   + '_' + \
                            'cons'    + format(args.consistency_weight, '.0E').replace('+', '')               + '_' + \
                            'tv'      + format(args.tv_weight, '.0E').replace('+', '')                        + \
                            ".pth"
        save_model_path = os.path.join(args.save_model_dir, save_model_filename)
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        torch.save(stylize_network.state_dict(), save_model_path)
        print("\nTrained model saved at ", save_model_path)


def main():

    # Arguments for training
    train_arg_parser = argparse.ArgumentParser(description="parser for fast neural style transfer")

    train_arg_parser.add_argument("--dataset", '--d', type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", '--i', type=str, required=True,
                                  help="path to style-image")
    train_arg_parser.add_argument("--model-type", type=str, default='ae',
                                  help="architecture for stylization network. including: 1. ae: Autoencoder; 2. "
                                       "bo: BottleNetwork; 3. res: ResNext; 4. dense: DenseNet")
    train_arg_parser.add_argument("--save-model-dir", type=str, default='./',
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--model-name", type=str, default=None,
                                  help="name for saved model. If not specified, \
                                  the model will be initialized same as style image.")

    train_arg_parser.add_argument("--content-weight", '--c', type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", '--s', type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--consistency-weight", '--cs', type=float, default=1e0,
                                  help="weight for consistency-loss, default is 1e0")
    train_arg_parser.add_argument("--tv-weight", '--tv', type=float, default=1e0,
                                  help="weight for tv-loss, default is 1e0")

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
