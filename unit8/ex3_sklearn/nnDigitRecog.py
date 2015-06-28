#! /usr/bin/env python
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# Given a training set of 20x20 pixel images and their corresponding
# digits, classify them using a neural network.
#


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import pandas as pd
from pandas import DataFrame
import h5py
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from Timer import Timer

sns.set_context('talk')

def plotRandImages(train_data, output):
    """
    Plot 4 random image of handwritten digits on one figure.
    """
    rint_array = np.random.randint(0,train_data.shape[0],size=4)

    fig = plt.figure(figsize=(8,8))
    plt.subplot(2,2,1)
    for ii,rint in enumerate(rint_array):
        plt.subplot(2,2,ii+1)
        img = train_data[rint].reshape((20,20)).T
        #print "  output: ",output[rint]
        plt.imshow(img,aspect='auto',interpolation='nearest',
                   origin='lower')
        plt.title('Image of %d'%output.T[rint])
    plt.show()

    return

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--data',metavar='HDF5',type=str,default='ex3data1.hdf5',
                    help='''Training dataset, taken from ex3data1.mat''')
parser.add_argument('--plot', action='store_true', default=False,
                    help='''Plotting features turned on.''')
args = parser.parse_args()

# Step 1: Load the training data
# a) As a 2D numpy array of 5000 rows 400 cols
# b) output as 1 row and 5000 cols
fh = h5py.File(args.data,'r')
train_X = np.array(fh['X']['value']).T
output_y = np.array(fh['y']['value'])
output_y = np.ravel(output_y)

# Step 2: Plot some random images:
if args.plot:
    plotRandImages(train_X,output_y)

# Step 3: Use artificial neural network for the training dataset.
# TODO: decide which nn library to use. Probably pylearn2?
#
