import os
import sys
import re

import torch
from torchvision import transforms
from torchvision.utils import save_image
import torch.onnx

import utils
from transformer_net import TransformerNet, TransformerNet_Atten


def check_paths(save_model_dir, checkpoint_model_dir):
    try:
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        if checkpoint_model_dir is not None and not (os.path.exists(checkpoint_model_dir)):
            os.makedirs(checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def stylize(cuda, content_image, content_scale, model, output_image, name):
    device = torch.device("cuda" if cuda else "cpu")

    content_image = utils.load_image(content_image, size=[1080, 1080], scale=content_scale)
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
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet_Atten()
        state_dict = torch.load(model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image).cpu()
        output = output_transform(output)
    # name = name.replace('.png', '.jpg')
    save_image(output[0], os.path.join(output_image, name))


cuda = 'cuda'
content_scale = 1


'''
This code is used for single photo stylizing
'''
model = 'D:/Project/PyPro/StyleTransfer/StyleTransfer/my_models/mosaic_c1E05_s1E10_p1E03_.pth'
img_dir = 'E:\project\Python\StyleTransfer/brick.jpg'
output_image = 'E:\project\Python\StyleTransfer'
stylize(cuda, img_dir, content_scale, model, output_image, 'fire.jpg')

'''
This code is used for multiple photo stylizing with one model
'''
# model = 'D:/Project/PyPro/StyleTransfer/StyleTransfer/my_models/inkwash_2.pth'
# img_dir = 'D:/Project/CPro/instant-ngp/data/nerf/real_world/mouatain_1/images'
# output_dir = 'D:/Project/CPro/instant-ngp/data/nerf/real_world/mouatain_1/style_images'
#
# dirlist = os.listdir(img_dir)
# for name in dirlist:
#     print(f'Painting: {name}')
#     content_image = os.path.join(img_dir, name)
#     stylize(cuda, content_image, content_scale, model, output_dir, name)

'''
This code is used for multiple photo stylizing with multiple model
'''
# model_dir = 'D:/Project/PyPro/StyleTransfer/StyleTransfer/my_models'
# model_list = os.listdir(model_dir)[6:]
#
# img_dir = 'D:/Project/PyPro/data/real-world'
# img_list = os.listdir(img_dir)
#
# output_folder_dir = 'D:/Project/PyPro/data/mosaic'
#
# for model in model_list:
#     folder_name = model.replace('_.pth', '')
#     output_folder = os.path.join(output_folder_dir, folder_name)
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     model = os.path.join(model_dir, model)
#
#     for name in img_list:
#         print(f'Painting: {name}')
#         content_image = os.path.join(img_dir, name)
#         stylize(cuda, content_image, content_scale, model, output_folder, name)
