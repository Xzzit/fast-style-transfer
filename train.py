import os
import time
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


def train(dataset, style_image, save_model_dir, epochs,
          content_weight=1e5, style_weight=1e10, consistency_weight=1e1, image_size=256, batch_size=16,
          model_name='inkwash', model='Autoencoder'):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # Define dataloader transform function
    content_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda a: a.mul(255))
    ])

    # Initialize dataloader
    train_dataset = CustomDataSet(dataset, content_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    # Initialize fast neural style transfer model and optimizer
    transformer = eval(model)().to(device)
    optimizer = Adam(transformer.parameters(), 1e-3)
    mse_loss = torch.nn.MSELoss()

    # Define pre-trained VGG net
    vgg = Vgg16(requires_grad=False).to(device)

    # Define transform function of style image loader
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    # Load style image
    style = utils.load_image(style_image, size=[image_size, image_size])
    style = style_transform(style)
    style = style.repeat(1, 1, 1, 1).to(device)

    # Compute style features
    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    # Train transformer net to match the style
    for e in range(epochs):

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

            count += batch_size
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            # Add noise to original image to increase the consistency
            noise = 1 / 10 * (torch.randn(x.shape) - 0.5)
            noise_x = x + noise.to(device)

            # Normalize batches
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
            consistency_loss = consistency_weight * mse_loss(noise_y, y)

            # Compute the total loss
            total_loss = content_loss + style_loss + consistency_loss
            total_loss.backward()
            optimizer.step()

            # Output the loss information
            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            agg_consistency_loss += consistency_loss.item()

            if (batch_id + 1) % 500 == 0:
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
        save_model_filename = model_name + '_' + \
                              'cont' + format(content_weight, '.0E').replace('+', '') + '_' + \
                              'sty' + format(style_weight, '.0E').replace('+', '') + '_' + \
                              'cons' + format(consistency_weight, '.0E').replace('+', '') + \
                              ".pth"
        save_model_path = os.path.join(save_model_dir, save_model_filename)
        torch.save(transformer.state_dict(), save_model_path)
        print("\nTrained model saved at ", save_model_path)


dataset = '/media/xzzit/APP/Project/PyPro/data/coco2017/train2017'
save_model_dir = 'pretrained_models/'
epochs = 1

"""
Train single model once a time.
"""
# style_image = 'pretrained_models/Fauvism_André-Derain_Pier.jpg'
# train(dataset, style_image, save_model_dir, epochs, 1e5, 1e10, 1e1)


"""
Train multiple models with different hyper-parameters.
"""
# style_image = 'D:/Project/PyPro/data/art/Claude_Monet_Le_Grand_Canal.jpg'
# for c, s, p in zip([1e5, 1e5, 1e6, 1e7, 1e7], [1e10, 1e11, 1e11, 1e11, 1e12], [1e2, 1e2, 1e2, 1e2, 1e2]):
#     train(dataset=dataset,
#           style_image=style_image,
#           save_model_dir=save_model_dir,
#           epochs=epochs,
#           consistency_weight=p,
#           style_weight=s,
#           content_weight=c)


'''
Train model with multiple style references
'''
# style_dir = 'D:/Project/PyPro/data/a'
# style_img_name = os.listdir(style_dir)

# for name in style_img_name:
#     style_image = os.path.join(style_dir, name)
#     train(dataset, style_image, save_model_dir, epochs,
#           content_weight=1e5, style_weight=1e10, consistency_weight=1e1, image_size=256, batch_size=8,
#           model_name=name.replace('.jpg', ''))


"""
Train multiple models.
"""
style_image = 'pretrained_models/Fauvism_André-Derain_Pier.jpg'
for m in ['Autoencoder', 'AutoencoderAttention', 'BottleNetwork', 'DenseNet', 'ResNext']:
    train(dataset=dataset,
          style_image=style_image,
          save_model_dir=save_model_dir,
          epochs=epochs,
          model=m,
          model_name=m)