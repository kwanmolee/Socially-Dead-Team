import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import torch
import numpy as np
from matplotlib.gridspec import GridSpec
import torchvision.transforms as transforms


def tic():
	return time.time()
def toc(tstart, name="Operation"):
	print('%s took: %s sec.\n' % (name,(time.time() - tstart)))

def GetName(paths):
	Name = []
	for p in paths:
		n = p.split("/")[-1].split(".")[0].split("_")
		new_n = " ".join(i.capitalize() for i in n if i != "demo")
		Name += [new_n]
	return Name

def Plot():
	style_dir = "SlowStyleTransfer/StyleImages"
	content_dir = "SlowStyleTransfer/ContentImages"
	output_dir = "SlowStyleTransfer/OutputImages"
	contents = [content_dir + "/" + filename for filename in os.listdir(content_dir)]
	styles = [style_dir  + "/" + filename for filename in os.listdir(style_dir) if filename != ".ipynb_checkpoints"]
	outputs = {}
	for c in os.listdir(style_dir):
		s = c.split(".")[0]
		outputs[s] = output_dir + "/{0}/".format(s)

	plt.figure(figsize=(50,50))
	gs = GridSpec(50, 50)
	s_names = GetName(styles)
	c_names = GetName(contents)

	for i in range(len(styles)+1):
		if i == 0:
			for c in range(4):
				plt.subplot(gs[i*10:i*10+10, c*10+10:c*10+20])
				im = Image.open(contents[c])
				im = im.resize((512,512))
				plt.imshow(im)
				plt.axis("off")
				plt.title("Content Image" + ": " + c_names[c], weight = "bold", fontsize = 40)
		else:
			for c in range(5):
				plt.subplot(gs[i*10:i*10+10, c*10:c*10+10])
				if c == 0:
					im = Image.open(styles[i-1])
					im = im.resize((512,512))
					plt.imshow(im)
					plt.axis("off")
					plt.title("Style Image" + ": " + s_names[i-1], weight = "bold", fontsize = 40)
				else:
					content = contents[c-1].split("/")[-1]
					extention = "demo_" + content
					path = outputs[styles[i-1].split("/")[-1].split(".")[0]] + extention
					im = Image.open(path)
					plt.imshow(im)
					plt.axis("off")
	plt.tight_layout()
	plt.show()

def LoadImage(img_name, device = "cuda"):
	im_size = 512 
	loader = transforms.Compose([transforms.Resize(im_size),  
									transforms.ToTensor()])  
	img = Image.open(img_name)
	img = loader(img)
	try:
		img = img.numpy()
	except:
		img = img.detach().numpy()
	img = np.moveaxis(img, [0, 1, 2], [2, 0, 1]) 
	return img

def display_example(style_name, content_name, output_name):
	style = "SlowStyleTransfer/StyleImages/{0}.jpg".format(style_name)
	content = "SlowStyleTransfer/ContentImages/{0}.jpg".format(content_name)
	output = "SlowStyleTransfer/OutputImages/{0}.jpg".format(output_name)
	fig, ax = plt.subplots(1,3, figsize = (15,15))
	#plt.suptitle("Display One Example for Slow Style transfer") 
	style_img = Image.open(style)
	style_img = style_img.resize((256,256))
	ax[0].imshow(style_img)
	ax[0].set_title("Style Image", weight = "bold", fontsize = 16)
	ax[0].axis('off')
	content_img = Image.open(content)
	content_img = content_img.resize((256,256))
	ax[1].imshow(content_img)
	ax[1].set_title("Content Image", weight = "bold", fontsize=16)
	ax[1].axis("off")
	opt = Image.open(output)
	opt = opt.resize((256,256))
	ax[2].imshow(opt)
	ax[2].set_title("Output Image", weight = "bold", fontsize = 16)
	ax[2].axis('off') 
	plt.show()
