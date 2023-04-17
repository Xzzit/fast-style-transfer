import os
import re

import torch
from torchvision import transforms
from torchvision.utils import save_image
import torch.onnx

import utils
from models import *


def stylize(content_image, model, output_image, name):

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    content_image = utils.load_image(content_image)
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
        style_model = AutoencoderOld()
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


'''
This code is used for single photo stylizing
'''
# model = 'pretrained_models/AutoencoderOld.pth'
# img_dir = 'pretrained_models/bear.jpg'
# output_image = './'
# stylize(img_dir, model, output_image, 'pretrained_models/1.jpg')

'''
This code is used for multiple photo stylizing with one model
'''
model = 'pretrained_models/AutoencoderOld.pth'
img_dir = '/media/xzzit/APP/Project/CPro/instant-ngp2023_4_17/data/nerf/real_world/mountain/mountain_1/baseline'
output_dir = 'pretrained_models/mountain_ae_old'

dirlist = os.listdir(img_dir)
for name in dirlist:
    print(f'Painting: {name}')
    content_image = os.path.join(img_dir, name)
    stylize(content_image, model, output_dir, name)

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


'''
This code is used for single photo stylizing with multiple models
'''
# model_dir = 'D:\Project\PyPro\StyleTransfer\Fast_Style_Transfer\pretrained_models/resNext'
# model_list = os.listdir(model_dir)

# img_dir = 'D:\Project\PyPro\StyleTransfer\Fast_Style_Transfer/bear.jpg'

# output_folder_dir = 'D:\Project\PyPro\StyleTransfer\Fast_Style_Transfer/resNext_outputs'

# for model in model_list:
#     start_name = model.replace('.pth', '')

#     model = os.path.join(model_dir, model)

#     stylize(img_dir, model, output_folder_dir, start_name+'.jpg')
