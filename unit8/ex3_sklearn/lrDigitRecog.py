#! /usr/bin/env python
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# Given a training set of 20x20 pixel images and their corresponding
# digits, classify them using logistic regression.
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
parser.add_argument('--reg', type=float, default=1.0e5,
                    help='''(Inverse) of the regularization parameter.''')
args = parser.parse_args()

# Step 1: Load the training data
# a) As a 2D numpy array of 5000 rows 400 cols
# b) output as 1 row and 5000 cols
fh = h5py.File(args.data,'r')
train_X = np.array(fh['X']['value']).T
output_y = np.array(fh['y']['value'])
output_y = np.ravel(output_y)

# b)As a pandas DataFrame
cols = ["pixel_%d"%ii for ii in xrange(train_X.shape[1])]
df = DataFrame(train_X, columns=cols)
print df.head()

# Step 2: Plot some random images:
if args.plot:
    plotRandImages(train_X,output_y)

print "X shape: ",np.shape(train_X)
print "y shape: ",np.shape(output_y)

# Step 3: Use multi-class classification to learn training sample.
# Fit a linear model using logistic regression

logreg = LogisticRegression(C=args.reg)
with Timer() as t:
    model = logreg.fit(train_X, output_y)
print("logistic regression took %s seconds to complete!"%t.secs)

print "Training Set Accuracy: ",model.score(train_X, output_y)

# Some benchmarks:
# For C = 10:    { accuracy: '96.5 %', time: '4.168 sec'  }
#     C = 1.0e5: { accuracy: '99.0 %', time: '104.44 sec' }
#
# Extending this project:
# Download the test dataset from MNIST database: http://yann.lecun.com/exdb/mnist/
#   And test the model on a test set not used in the training to see how
#   generalizable the model is.
#
