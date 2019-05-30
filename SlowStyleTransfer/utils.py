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

def Plot(content_dir, style_dir, output_dir):
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
				im = LoadImage(contents[c], "cuda")
				plt.imshow(im)
				plt.axis("off")
				plt.title("Content Image" + ": " + c_names[c], weight = "bold", fontsize = 40)
		else:
			for c in range(5):
				plt.subplot(gs[i*10:i*10+10, c*10:c*10+10])
				if c == 0:
					im = LoadImage(styles[i-1], "cuda")
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
