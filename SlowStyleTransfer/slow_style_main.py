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
    main_arg_parser = argparse.ArgumentParser(description="Slow-Style-Transfer by Socially-Dead-Team ")
    #initialize arguments parser
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")
    sst_arg_parser = subparsers.add_parser("sst", help="personalized arguments for model buidling, training and output")

    sst_arg_parser.add_argument("--style_image", type=str, 
                                    help="directory of style image")
    sst_arg_parser.add_argument("--content_image", type=str, 
                                    help="directory of content image")
    sst_arg_parser.add_argument("--output_image", type=str, 
                                    help="directory of output image")

    sst_arg_parser.add_argument("--epochs", type=int, default=100,
                                    help="number of epochs")
    # training parameters
    sst_arg_parser.add_argument("--style_weight", type=int, default=320000,
                                    help="the style weight")
    sst_arg_parser.add_argument("--content_weight", type=str, default=1,
                                    help="the content weight")

    # mode transform
    sst_arg_parser.add_argument("--saved_model", type=str,default=None,
                                    help="saved training model per style target")
    
    # output choices
    sst_arg_parser.add_argument("--print_freq", type=int, default=100,
                                    help="print frequency")
    sst_arg_parser.add_argument("--checkpoint_path", type=str, default = None,
                                    help="directory of checkpoints")
    
    # test mode
    sst_arg_parser.add_argument("--test_mode", type=bool, default=False,
                                    help="mode for evaluation on test image")
    sst_arg_parser.add_argument("--content_name", type=str, default = None,
                                    help="name of content images")    
    sst_arg_parser.add_argument("--style_name", type=str, default = None,
                                    help="name of style images")  
    sst_arg_parser.add_argument("--output_name", type=str, default = None,
                                    help="name of output images")  
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
