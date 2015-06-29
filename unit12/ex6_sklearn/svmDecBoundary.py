#! /usr/bin/env python
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# Given a dataset of labelled points, plot the points along with their
# label (true or false).
#
# NEXT: Use sklearn.svm.SVC to classify the points and learn the
# decision boudary. THIS SCRIPT SHOULD BE USED more as an interactive
# tool to modify the inputs to SVM in order to experiment with the
# decision boundary and learning.
#


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys

import numpy as np
import pandas as pd
from pandas import DataFrame
import h5py
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
sys.path.append('../../common')
from Timer import Timer

sns.set_context('talk')

def plotTrainingData(df):
    """
    Plots training data along with their label, and returns axes.
    """

    df_true = df[df['output']==1]
    df_false = df[df['output']==0]

    sns.regplot("x1","x2", df_true, marker="x", fit_reg=False,
                scatter_kws={"color": "blue", "s": 50},
                label="True")
    ax = plt.gca()
    sns.regplot("x1", "x2", df_false, marker="o", fit_reg=False, ax=ax,
                scatter_kws={"color": "red", "s": 50},
                label="False")

    plt.legend(loc='best',frameon=True,framealpha=0.7)

    return plt.gca()


def plotDecBoundary(clf, df, ax):
    """
    Plots decision boundary as a translucent image on top of the points.
    """

    npts = 101
    x1 = np.linspace(df['x1'].min(), df['x1'].max(), npts)
    x2 = np.linspace(df['x2'].min(), df['x2'].max(), npts)

    xx1, xx2 = np.meshgrid(x1,x2)
    pairs = np.column_stack((xx1.ravel(), xx2.ravel()))

    img = clf.predict(pairs)
    img = img.reshape((npts,npts))

    ax.imshow(img,extent=[np.min(x1),np.max(x1),np.min(x2),np.max(x2)],
              interpolation='nearest', origin='lower',
              cmap=plt.cm.Paired_r,alpha=0.4)


    return plt.gca()


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('data',metavar='HDF5',type=str,
                    help='''Training dataset, taken from ex3data1.mat''')
args  = parser.parse_args()

# Step 1: Load the training data
fh = h5py.File(args.data,'r')
train_X = np.array(fh['X']['value']).T
output_y = np.array(fh['y']['value'])
output_y = np.ravel(output_y)

# Load into DataFrame:
df = DataFrame(np.column_stack((train_X, output_y)),
               columns=['x1','x2','output'])

# Plot training dat
fig = plotTrainingData(df)


#
# Step 2: Initialize SVC Here is where you play around with a bunch of
# different parameters of the SVC including "kernel", "C", "degree",
# "gamma", etc.
#
clf = SVC(C=1.0,gamma=100.0,kernel='rbf')
with Timer() as t:
    clf.fit(train_X, output_y)
print("   ==> SVC took %s seconds to finish"%t.secs)


# Step 3: Plot Decision Boundary
fig2 = plotDecBoundary(clf, df, fig)

plt.show()
