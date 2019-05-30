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

from slow_style_utils import *
from slow_style_transfer import *

def main():
	main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
	#initialize arguments parser
	subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")
	stylize_arg_parser = subparsers.add_parser("stylize", help="parser for training arguments")

	#add style and content image arguments
	stylize_arg_parser.add_argument("--style_image", type=str, required=True,
	                                help="directory of style-image")
	stylize_arg_parser.add_argument("--content_image", type=str, required=True,
	                                help="directory of content-image")
	#add output image arguments
	stylize_arg_parser.add_argument("--output_image", type=str, required=True,
	                             	help="directory of output-image")
	stylize_arg_parser.add_argument("--num_steps", type=int, default=300,
	                             	help="number of iterations")
	#add style and content weights
	stylize_arg_parser.add_argument("--style_weight", type=str, default=1000000,
	                             	help="the style weight")
	stylize_arg_parser.add_argument("--content_weight", type=str, default=1,
	                             	help="the content weight")


	args = main_arg_parser.parse_args()


	if args.subcommand is None:
	    print("ERROR: specify the stylize parameters")
	    sys.exit(1)
	#implete slow style transfer
	sst = SlowStyleTransfer()
	sst.ImplementTransferLearning(args)
	plt.show()


if __name__ == "__main__":
    main()
