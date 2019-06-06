# Socially-Dead-Team
[![Project Team Status](https://img.shields.io/badge/Socially%20Dead%20Team-Style%20Transfer%20Learning-lightcoral.svg)](https://github.com/oliver-lijiayi/Socially-Dead-Team)
[![Project 1 Status](https://img.shields.io/badge/1-Slow%20Style%20Transfer-lightskyblue.svg)](https://github.com/oliver-lijiayi/Socially-Dead-Team/tree/master/SlowStyleTransfer)
[![Project 2 Status](https://img.shields.io/badge/2-Fast%20Style%20Transfer-7777aa.svg)](https://github.com/oliver-lijiayi/Socially-Dead-Team/tree/master/FastStyleTransfer)
[![Result Status](https://img.shields.io/badge/3-Results-mediumaquamarine.svg)](https://github.com/oliver-lijiayi/Socially-Dead-Team/blob/master/style_transfer_learning.ipynb)
[![Retraining Status](https://img.shields.io/badge/4-Re--taining-orange.svg)](https://github.com/oliver-lijiayi/Socially-Dead-Team/blob/master/fast_style_transfer_training.ipynb)

## Content Overview
* [Prerequisites](#Prerequisites)
* [Code Organization](#Code-Organization)
* [Dataset](#Dataset)
* [Slow Style Transfer](#Slow-Style-Transfer)
* [Fast Style Transfer](#Fast-Style-Transfer)
* [Examples](#Examples)
   * [Demonstration of Reconstruction with White Noise Image](#Demonstration-of-Reconstruction-with-White-Noise-Image)
   * [Style Images](#Style-Images)
   * [Cotent Images](#Cotent-Images)
   * [Result Images](#Result-Images)
   	* [Slow Style Transfer Results](#Slow-Style-Transfer-Results)
   	* [Fast Style Transfer Results](#Fast-Style-Transfer-Results)
   

## Prerequisites
<pre>
1. pytorch: 3.6.4
2. NumPy: 1.14.2
3. matplotlib: 2.2.0
4. Pillow: 5.1.0
5. cv2: 4.0.0
6. GPU: NVIDIA GPU is advised
</pre>

## Code Organization
<pre>
<b>SlowStyleTransfer</b><br>
  1. slow_style_utils.py: some utility functions and classes 
  2. slow_style_transfer.py: the class for the implementation of slow style transfer
  3. slow_style_main.py: get code run 
  4. utils.py: some utility functions making style_transfer_learning.ipynb clean

<b>FaststyleTransfer</b><br>
  1. utils.py: helper functions 
  2. fast_style_transfer.py: main function package that contains all functions for this project
  3. transformer_net.py: class for the transform network 
  4. vgg.py: class for the vgg loss network

<b>style_transfer_learning.ipynb</b>
	For demonstrating the results of this project.
<b>fast_style_training.ipynb</b>
	For training style models for fast style transfer implementation.
</pre>

## Dataset
All data from [COCO-2015](http://cocodataset.org/#home)

## Demonstration of Reconstruction with White Noise Image



## Slow Style Transfer
This is the implementation of [Gatys method](https://arxiv.org/pdf/1508.06576.pdf) on Neural Style Transfer.
For rerunning the slow style transfer, please go to [the demonstration file](https://github.com/oliver-lijiayi/Socially-Dead-Team/blob/master/style_transfer_learning.ipynb) 
<p>
 <img src="https://github.com/oliver-lijiayi/Socially-Dead-Team/blob/master/SlowStyleTransfer/OutputImages/result1.jpg" width="200"/>
  <img src="https://github.com/oliver-lijiayi/Socially-Dead-Team/blob/master/SlowStyleTransfer/OutputImages/result2.jpg" width="200"/>	
</p>

## Fast Style Transfer 
Related architecture and techniques are introduced in [this paper](https://arxiv.org/pdf/1508.06576.pdf).
The main structure we utilize is demonstrated below
![model](https://raw.githubusercontent.com/kwanmolee/-Style-Transfer-Learning/master/model.png)

## Examples

### Style Images
<p float="left">
  <img src="https://github.com/oliver-lijiayi/Socially-Dead-Team/blob/master/SlowStyleTransfer/StyleImages/cat.jpg" width="150"/>
  <img src="https://github.com/oliver-lijiayi/Socially-Dead-Team/blob/master/SlowStyleTransfer/StyleImages/comic.jpg" width="200"/>
  <img src="https://github.com/oliver-lijiayi/Socially-Dead-Team/blob/master/SlowStyleTransfer/StyleImages/mosaic.jpg" width="200"/>
  <img src="https://github.com/oliver-lijiayi/Socially-Dead-Team/blob/master/SlowStyleTransfer/StyleImages/picasso.jpg" width="200"/>
</p>

### Cotent Images
<p float="left">
  <img src="https://github.com/oliver-lijiayi/Socially-Dead-Team/blob/master/SlowStyleTransfer/ContentImages/amber.jpg" width="180"/>
  <img src="https://github.com/oliver-lijiayi/Socially-Dead-Team/blob/master/SlowStyleTransfer/ContentImages/geisel.jpg" width="280"/>
  <img src="https://github.com/oliver-lijiayi/Socially-Dead-Team/blob/master/SlowStyleTransfer/ContentImages/bear.jpg" width="180"/>
  <img src="https://github.com/oliver-lijiayi/Socially-Dead-Team/blob/master/SlowStyleTransfer/ContentImages/house.jpg" width="200"/>
</p>

### Result Images

#### Slow Style Transfer Results
<p float="left">
  <img src="https://github.com/oliver-lijiayi/Socially-Dead-Team/blob/master/SlowStyleTransfer/OutputImages/SST_result.jpg" width="1000"/>
</p>

#### Fast Style Transfer Results
<p float="left">
  <img src="https://github.com/oliver-lijiayi/Socially-Dead-Team/blob/master/FastStyleTransfer/output_images/grid%20result.png" width="1000"/>
</p>


	
