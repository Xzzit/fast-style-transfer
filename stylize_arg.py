import os
import re
import argparse
import sys

import torch
from torchvision import transforms
from torchvision.utils import save_image
import torch.onnx

import utils
from models import *


def stylize(args):

    # Set device
    if args.mps:
        device = torch.device('mps')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load content image
    content_image = utils.load_image(args.content_image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        # mean=[0.40760392, 0.45795686, 0.48501961], std=[1, 1, 1]
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1]),
        transforms.Lambda(lambda x: x.mul_(255))
    ])
    output_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.mul_(1. / 255)),
        # transforms.Normalize(mean=[-0.185, -0.156, -0.106], std=[1, 1, 1]),
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)

    with torch.no_grad():

        # Initialize fast neural style transfer model
        if args.model_type == 'ae':
            style_model = Autoencoder().to(device)
        elif args.model_type == 'bo':
            style_model = BottleNetwork().to(device)
        elif args.model_type == 'res':
            style_model = ResNext().to(device)
        else:
            print('Error: invalid selected architecture')
            sys.exit()

        state_dict = torch.load(args.model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image).cpu()
        output = output_transform(output)
    # name = name.replace('.png', '.jpg')
    save_image(output[0], os.path.join(args.output_path, args.output_name))


def main():
    # Arguments for stylizing
    eval_arg_parser = argparse.ArgumentParser(description="parser for fast neural style transfer")

    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image")
    eval_arg_parser.add_argument("--output-path", type=str, default='./',
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--output-name", type=str, default='stylized.jpg',
                                 help="name of the stylized image")

    eval_arg_parser.add_argument("--model-type", type=str, default='ae',
                                 help="architecture for stylization network. including: 1. ae: Autoencoder; 2. "
                                      "bo: bottleneck; 3. res: resNext")

    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")

    eval_arg_parser.add_argument('--mps', action='store_true', default=False, help='enable macOS GPU stylizing')

    args = eval_arg_parser.parse_args()
    stylize(args)


if __name__ == '__main__':
    main()
