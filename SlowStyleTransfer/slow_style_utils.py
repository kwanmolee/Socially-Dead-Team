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
	# load size based on whether gpu is available
	im_size = 512 if torch.cuda.is_available() else 128  
	loader = transforms.Compose([transforms.Resize(im_size),  
									transforms.ToTensor()])  
	img = Image.open(img_name)
	img = loader(img).unsqueeze(0)
	img = img[:,:,:im_size,:im_size]
	return img.to(device, torch.float)

def SaveImage(tensor, args):
	# transfer, clone and remove fake dimension
	unloader = transforms.ToPILImage()  
	img = tensor.cpu().clone()  
	img = img.squeeze(0)      
	img = unloader(img)
	if args.test_mode == True:
		opt_path = "SlowStyleTransfer/OutputImages/{0}.jpg".format(args.output_name)
		img.save(opt_path)
	else:
		img.save(args.output_image)

def PlotImage(args):
	# plot image
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
	# detach content from image
	def __init__(self, target,):
		super(ContentLoss, self).__init__()
		self.target = target.detach()

	def forward(self, input):
		# throw error if content is a variable
		self.loss = F.mse_loss(input, self.target)
		return input

class StyleLoss(nn.Module):
	# detach style from image
	def __init__(self, target_feature):
		super(StyleLoss, self).__init__()
		self.target = self.gram_matrix(target_feature).detach()

	def forward(self, input):
		G = self.gram_matrix(input)
		self.loss = F.mse_loss(G, self.target)
		return input

	def gram_matrix(self, input):
		# transfer iamge into gram matrix
		a, b, c, d = input.size()  
		features = input.view(a * b, c * d)  
		# compute the gram product
		G = torch.mm(features, features.t())  
		# normalize the values of G
		return G.div(a * b * c * d)

class Normalization(nn.Module):
	# normalize image to fit the Tensor
	def __init__(self, mean, std):
		super(Normalization, self).__init__()
		self.mean = mean.clone().detach().view(-1, 1, 1)
		self.std = std.clone().detach().view(-1, 1, 1)

	def forward(self, img):
		return (img - self.mean) / self.std
