#! /usr/bin/env python
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# Linear regression with multiple variables, example taken from
# Coursera Machine Learning course. (Does not include any cross
# validation or regularization).
#
#

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pandas as pd
from pandas import DataFrame
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

sns.set_context('talk')


def plotCorrelation(frame):

    # Plot correlation of each variable to visualize each dimension:
    sns.jointplot("bedrooms","price",frame,size=8)
    plt.tight_layout()
    sns.jointplot("size","price",frame,size=8)
    plt.tight_layout()

    print("PAUSED...close figures to continue...")
    plt.show()
    return

parser = ArgumentParser(
    '''Predict Fair Housing prices from training set in Portland, Oregon,
    using features of size and bedrooms.''')
parser.add_argument('--data',type=str,default='ex1data2.txt',
                    help='''Input data file for the housing price prediction.''')
parser.add_argument('--plot',action='store_true',default=False,
                    help='''Turn plotting features on.''')
args = parser.parse_args()


# Step 1: read in data and prepare for training
df = pd.read_csv(args.data,names=['size','bedrooms','price'])

# rows = training ex, cols = features
bedrooms = np.array(df['bedrooms']); size = np.array(df['size'])
features_X = np.column_stack((size,bedrooms))
#print features_X

output_y = np.array(df['price'])

# Step 2: make scatter plot of each feature
if args.plot: plotCorrelation(df)

# Step 3: Fit the linear regression model
regr = LinearRegression()
regr.fit(features_X, output_y)

print("intercept: ",regr.intercept_)
print("coefficient: ",regr.coef_)

# regression score:
print("Variance score: %.2f"%regr.score(features_X, output_y))

# How much would a 3 bedroom, 1650 sq foot home cost?

print "  -->Price for: ", regr.predict([[1650,3]])
