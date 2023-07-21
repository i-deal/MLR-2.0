#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:04:16 2020

@author: brad
"""
# prerequisites
import torch
import numpy as np
from sklearn import svm
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.optim as optim
import config 

config.init()

from config import numcolors, args
global numcolors
from mVAE import train, test, vae,  thecolorlabels, optimizer, dataset_builder

if torch.cuda.is_available():
    device = 'cuda'
    print('CUDA')
else:
    device = 'cpu'

# reload a saved file
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,device)
    vae.load_state_dict(checkpoint['state_dict'])
    #for parameter in vae.parameters():
        #parameter.requires_grad = False
    #vae.eval()
    return vae

load_checkpoint('output{modelNumber}/checkpoint_threeloss_singlegrad200_smfc.pth'.format(modelNumber=1)) #temp
#define color labels 
#this list of colors is randomly generated at the start of each epoch (down below)

#numcolors indicates where we are in the color sequence 
#this index value is reset back to 0 at the start of each epoch (down below)
numcolors = 0
#this is the amount of variability in r,g,b color values (+/- this amount from the baseline)

#these define the R,G,B color values for each of the 10 colors.  
#Values near the boundaries of 0 and 1 are specified with colorrange to avoid the clipping the random values
folder_path = f'output'

if not os.path.exists(folder_path):
    os.mkdir(folder_path)

data_set_flag ='padded_mnist'
train_loader_noSkip, train_loader_skip, test_loader_noSkip, test_loader_skip = dataset_builder(data_set_flag)

for epoch in range(1, 201):
    #modified to include color labels
    train(epoch,'iterated', train_loader_noSkip, train_loader_skip)
    colorlabels = np.random.randint(0,10,100000)#regenerate the list of color labels at the start of each test epoch
    numcolors = 0
    #if epoch % 5 == 0:
     #   test('all')  
   
    if epoch in [1,25,50,75,100,150,200,300,400,500]:
        checkpoint =  {
                 'state_dict': vae.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                      }
        torch.save(checkpoint,f'{folder_path}/checkpoint_threeloss_singlegrad{str(epoch)}_smfc.pth')






