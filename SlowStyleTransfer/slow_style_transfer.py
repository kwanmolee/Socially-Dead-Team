""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
-----------------------   moduele import ------------------------------

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import sys
import copy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
from slow_style_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
-----------------------   slow style Transfer  -----------------------

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class SlowStyleTransfer():

    def __init__(self):
        pass

    def BuildStyleModel(self, cnn, normalization_mean, 
                                normalization_std,
                                style_img, content_img,
                                content_layers=None,
                                style_layers=None):
        cnn = copy.deepcopy(cnn)
        normalization = Normalization(normalization_mean, normalization_std).to(device)
        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)
        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)

            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)

            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)

            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)

            else:
                raise RuntimeError('Unrecognized Layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)
            if name in content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]
        return model, style_losses, content_losses

    def InputOptimizer(self,input_img):
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer


    def ImplementTransferLearning(self, args):
        if args.test_mode == True:
            c_path = "SlowStyleTransfer/ContentImages/{0}.jpg".format(args.content_name)
            s_path = "SlowStyleTransfer/StyleImages/{0}.jpg".format(args.style_name)
            content_img = LoadImage(c_path, device)
            style_img = LoadImage(s_path, device)
            input_img = content_img.clone()
        else:  
            content_img = LoadImage(args.content_image, device)
            style_img = LoadImage(args.style_image, device)
            input_img = content_img.clone()
        epochs = args.epochs
        style_weight = args.style_weight
        content_weight = args.content_weight
              
        optimizer = self.InputOptimizer(input_img)
        
        if args.saved_model is None:
            cnn = models.vgg19(pretrained = True).features.to(device).eval()
            cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
            cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
            content_layers_default = ['conv_4']
            style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
            print('1. Building Style Transfer Model\n')
            model, style_losses, content_losses = self.BuildStyleModel(cnn,\
                cnn_normalization_mean, cnn_normalization_std, style_img, content_img,\
                content_layers_default, style_layers_default)
            print('2. Start Optimization\n')
        else:
            checkpoint = torch.load(args.saved_model)
            model = checkpoint['model']
            style_losses = checkpoint['style_losses']
            content_losses = checkpoint['content_losses']
                       
        Epochs = [0]
        while Epochs[0] <= epochs:
            def closure():
                # correct the values of updated input image
                input_img.data.clamp_(0, 1)
                
                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0
                
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                
                style_score *= style_weight
                content_score *= content_weight
                
                loss = style_score + content_score
                loss.backward()
                
                Epochs[0] += 1
                if args.print_freq > 0:
                    if Epochs[0] == 1 or Epochs[0] % 100 == 0:
                            print("\tEpochs {0}:\n\tStyle Loss : {1:.4f}\tContent Loss: {2:.4f}\n".\
                            format(Epochs[0],style_score.item(),content_score.item()))
                return style_score + content_score

            optimizer.step(closure)
            self.save_checkpoint(Epochs, model, style_losses, content_losses,\
                                 args, filename = "checkpoint.pth.tar")

        # a last correction...
        input_img.data.clamp_(0, 1)
        SaveImage(input_img, args)
        #PlotImage(args)
        plt.show()


    def save_checkpoint(self, epochs, model, style_losses, content_losses, \
                            args, filename = "sst_checkpoint.pth.tar"):
        root = args.checkpoint_path
        if args.test_mode == True:
            prefix = args.style_name
        else:
            prefix = args.style_image.split(".")[0].split("/")[-1]
        if root is not None:
            torch.save({
            'epoch': epochs,
            'model': model,
            'style_losses': style_losses,
            'content_losses': content_losses,
            }, "{0}/{1}/{1}_{2}".format(root, prefix, filename))
        else:
            torch.save({
            'epoch': epochs,
            'model': model,
            'style_losses': style_losses,
            'content_losses': content_losses,
            }, "{0}_{1}".format(prefix, filename))
    
    
        
        
