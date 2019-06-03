#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import utils
import matplotlib.pyplot as plt
from PIL import Image
from transformer_net import TransformerNet
from vgg import Vgg16

import cv2


def train(style_img,
      epochs = 2, 
      batch_size = 4, 
      dataset = "/datasets/COCO-2015",
      style_size = None,
      save_model_dir = os.path.join(os.getcwd(),"FastStyleTransfer/saved_models"),
      checkpoint_model_dir = os.path.join(os.getcwd(), "FastStyleTransfer/checkpoints"),
      image_size = 256,
      random_seed = 42,
      content_weight = 1e5,
      style_weight = 1e10,
      lr = 1e-3,
      log_interval = 500):
    '''
    train the network for a selected style
    style_img: path to the selected style image
    save_model_dir: directory where the trained models would be stored
    checkpoint_model_dir: directory where the checkpoint during training would be stored
    '''
    _, tail = os.path.split(style_img)
    style_name = tail.split('.')[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # cwd = os.getcwd()
    save_model_filename = style_name + ".pth"
    save_model_path = os.path.join(save_model_dir, save_model_filename)
    
    try: #check whether the output directory is valid
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        if checkpoint_model_dir is not None and not (os.path.exists(checkpoint_model_dir)):
            os.makedirs(checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)
    if os.path.isfile(save_model_path):#model already exist
        print("model for this style has already been trained")
        return
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory = True, shuffle = True)
    
    # Image Transform Network 
    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), lr)
    mse_loss = torch.nn.MSELoss()
    
    # Style transform Network
    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(style_img, size=style_size)
    style = style_transform(style)
    style = style.repeat(batch_size, 1, 1, 1).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]
    
    ckpt_model_filename = "ckpt_" + style_name + ".pth"
    ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
    start_epoch = 0
    if os.path.isfile(ckpt_model_path): #checkpoint existed
        checkpoint = torch.load(ckpt_model_path, map_location=device)
        transformer.load_state_dict(checkpoint)
        start_epoch = 1
        print("a checkpoint is recovered :)")
        del checkpoint
    for e in range(start_epoch, epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)
            if (batch_id + 1) > 80000: #only use the first 80000 samples from the dataset
                break
                
        # save the checkpoint
        if checkpoint_model_dir is not None:
            transformer.eval().cpu()
            torch.save(transformer.state_dict(), ckpt_model_path)
            transformer.to(device).train()

    # save model
    transformer.eval().cpu()
    
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)



def evaluate (content = 'arch', style = 'mosaic', content_scale = None,output_dir = os.path.join(os.getcwd(), "FastStyleTransfer/output_images")):
    '''
    generate an image with given content and style
    content: name of the content image
    style: name of style wanted
    output_dir: directory where generated image is stored
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except OSError as e:
        print(e)
        sys.exit(1)
    cwd = os.getcwd()
    content_img_path = os.path.join(cwd, f"FastStyleTransfer/content_images/{content}.jpg")
    if not os.path.isfile(content_img_path):
        print("invalid content image path")
        return
    style_model_path = os.path.join(cwd, f"FastStyleTransfer/saved_models/{style}.pth")
    if not os.path.isfile(style_model_path):
        print("invalid style model path")
        return
    style_img_path = os.path.join(cwd, f"FastStyleTransfer/style_images/{style}.jpg")
    if not os.path.isfile(style_img_path):
        print("invalid style image path")
        return 
    _, tail = os.path.split(content_img_path)   
    content_name = tail.split('.')[0]
    _, tail = os.path.split(style_model_path)
    style = tail.split('.')[0]
    content_image = utils.load_image(content_img_path,size = 256, scale=content_scale)
    content = content_image
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(style_model_path)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image).cpu()
    output_path = os.path.join(output_dir, content_name + "-" + style + ".jpg")
    utils.save_image(output_path, output[0])
    
    #display result
    fig, ax = plt.subplots(1,3, figsize = (15,15))
    ax[0].imshow(content)
    ax[0].set_title('Content Image', fontsize=15)
    ax[0].axis("off")
    style_img = Image.open(style_img_path)
    style_img = style_img.resize((256,256))
    ax[1].imshow(style_img)
    ax[1].set_title("Style Image", fontsize = 15)
    ax[1].axis('off')
    new_image = Image.open(output_path)
    new_image = new_image.resize((256,256))
    ax[2].imshow(new_image)
    ax[2].set_title("After Style Transfer", fontsize = 15)
    ax[2].axis('off')
#     plt.tight_layout()



def showresult(style_list = ["cat","comic","mosaic","picasso"], content_list = ["arch","bear","geisel","house"]):
    '''
    show result of style transfer
    style_list: a list of style name
    content_list: a list of content images' names
    '''
    height = len(style_list) + 1
    width = len(content_list) + 1
    fig,ax = plt.subplots(height, width, figsize = (height * 4, width * 4))
    cwd = os.getcwd()
    content_dir = os.path.join(cwd, "FastStyleTransfer/content_images")
    stylized_dir = os.path.join(cwd,"FastStyleTransfer/output_images")
    style_dir = os.path.join(cwd,"FastStyleTransfer/style_images")
    
    for axi in ax.ravel():
        axi.axis('off')
    
    for index in range(len(style_list)):
        style = style_list[index]
        style_path = os.path.join(style_dir, style + ".jpg")
        style_img = Image.open(style_path)
        style_img = style_img.resize((256,256))
        ax[index + 1,0].imshow(style_img)

    for index in range(len(content_list)):    
        content = content_list[index]
        content_path = os.path.join(content_dir,content + ".jpg" )
        content_img = Image.open(content_path)
        content_img = content_img.resize((256,256))
        ax[0, index + 1].imshow(content_img)

        row = 1
        for style in style_list:
            content_style = content + '-' + style
            stylized_path = os.path.join(stylized_dir, content_style + ".jpg")
            stylized_img = Image.open(stylized_path)
            stylized_img = stylized_img.resize((256,256))
            ax[row, index + 1].imshow(stylized_img)
            row += 1

    fig.subplots_adjust(hspace = 0.01, wspace = 0.005)



def transform(x, style_model, device):
    content_image = x
    content_image = Image.fromarray(content_image)

    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = style_model(content_image).cpu()
    img = output[0].clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    return img



def realtime_stylize(styles = ["mosaic", "comic", "picasso"]):
    '''
    show real time style transfer with selected three styles and the origin image at the same time
    styles: a list of selected styles (length: 3)
    '''
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
#    device = 'cpu'
    model_dir = os.path.join(os.getcwd(),"FastStyleTransfer/saved_models")
    # init webcam
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    # create 3 models
    style_models = list()
    for style in styles:
        model_path = os.path.join(model_dir,style + ".pth")
        if not os.path.isfile(model_path):
            print(f"selected model ({style}) does not exist")
            return
        # style_models.append(TransformerNet())
        # style_models[-1].load_state_dict(torch.load(model_path))
        # style_models[-1].to(device)

        style_model = TransformerNet()
        state_dict = torch.load(model_path)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_models.append(style_model)

    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Use lower size image if running on cpu
        if device == 'cpu':
            frame = cv2.resize(frame, (160,120), interpolation = cv2.INTER_AREA)

        # forward 3 style transforms here
        frame_styles = list()
        for style_model in style_models:
            frame_styles.append(transform(frame,style_model,device))

        # combine 3 stylized image with the raw image
        frame_combined1 = np.concatenate((frame,frame_styles[0]),axis = 1)
        frame_combined2 = np.concatenate((frame_styles[1], frame_styles[2]), axis=1)
        frame_combined = np.concatenate((frame_combined1,frame_combined2),axis = 0)

        out.write(frame_combined)
        # Display the resulting frame
        cv2.imshow('Realtime stylize', frame_combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
