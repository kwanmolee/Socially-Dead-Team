import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import argparse
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import copy

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
-----------------------   img Processing  -----------------------

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def LoadImage(img_name, device):
	im_size = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
	loader = transforms.Compose([transforms.Resize(im_size),  
									transforms.ToTensor()])  
	img = Image.open(img_name)
	img = loader(img).unsqueeze(0)
	img = img[:,:,:im_size,:im_size]
	return img.to(device, torch.float)

def SaveImage(tensor, args):
	unloader = transforms.ToPILImage()  # reconvert into PIL img
	img = tensor.cpu().clone()  # we clone the tensor to not do changes on it
	img = img.squeeze(0)      # remove the fake batch dimension
	img = unloader(img)
	img.save(args.output_image)

def PlotImage(args):
	plt.figure(figsize=(15,15))
	paths = [args.style_image, args.content_image, args.output_image]
	title = ["Style Image", "Content Image", "Output Image"]
	n = 131
	for i in range(len(paths)):
		plt.subplot(n)
		im = Image.open(paths[i])
		plt.imshow(im)
		plt.xticks([])
		plt.yticks([])
		plt.title(title[i])
		n += 1
	plt.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
-----------------------   Utilitty Classes  -----------------------

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        

class ContentLoss(nn.Module):
	def __init__(self, target,):
		super(ContentLoss, self).__init__()
		self.target = target.detach()

	def forward(self, input):
		self.loss = F.mse_loss(input, self.target)
		return input

class StyleLoss(nn.Module):
	def __init__(self, target_feature):
		super(StyleLoss, self).__init__()
		self.target = self.gram_matrix(target_feature).detach()

	def forward(self, input):
		G = self.gram_matrix(input)
		self.loss = F.mse_loss(G, self.target)
		return input

	def gram_matrix(self, input):
		a, b, c, d = input.size()  # a=batch size(=1)
		# b=number of feature maps
		# (c,d)=dimensions of a f. map (N=c*d)

		features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
		G = torch.mm(features, features.t())  # compute the gram product

		# we 'normalize' the values of the gram matrix
		# by dividing by the number of element in each feature maps.
		return G.div(a * b * c * d)

class Normalization(nn.Module):
	def __init__(self, mean, std):
		super(Normalization, self).__init__()
		self.mean = mean.clone().detach().view(-1, 1, 1)
		self.std = std.clone().detach().view(-1, 1, 1)

	def forward(self, img):
		return (img - self.mean) / self.std