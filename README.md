# Socially-Dead-Team
[![Project Overview Status](https://img.shields.io/badge/Socially%20Dead%20Team-Style%20Transfer%20Learning-red.svg)](https://github.com/oliver-lijiayi/Socially-Dead-Team)

## Content Overview
* [Prerequisites](#Prerequisites)
* [Code Organization](#Code-Organization)
* [Slow Style Transfer](#Slow-Style-Transfer)
   * [1](#1)
* [Fast Style Transfer](#Fast-Style-Transfer)

## Prerequisites
<pre>
- pytorch: 3.6.4
- NumPy: 1.14.2
- matplotlib: 2.2.0
- Pillow: 5.1.0
- cv2:
- GPU: NVIDIA GPU is advised
</pre>

## Code Organization
<pre>
<b>SlowStyleTransfer</b><br>
  1. slow_style_utils.py: some utility functions and classes 
  2. slow_style_transfer.py: the class for the implementation of slow style transfer
  3. slow_style_main.py: get code run 
  4. utils.py: some utility functions making style_transfer_learning.ipynb clean
  <b>FaststyleTransfer</b><br>
  1. utils.py: 
  2. fast_style_transfer.py: 
  3. transformer_net.py:  
  4. neural_style.py: 
</pre>

## Slow Style Transfer
This is the implementation of [Gatys method](https://arxiv.org/pdf/1508.06576.pdf) on Neural Style Transfer.

### 1

## Fast Style Transfer 
Related architecture and techniques are introduced in [this paper](https://arxiv.org/pdf/1508.06576.pdf).
The main structure we utilize is demonstrated below
![model](https://raw.githubusercontent.com/kwanmolee/-Style-Transfer-Learning/master/model.png)



	
