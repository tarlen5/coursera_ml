#! /usr/bin/env python
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# Predict admission rates based on two exam scores using a Logistic
# Regression model. We also answer some questions about the data such as:
#   1. Given exam scores of (x1,x2) how likely is it to be admitted?
#   2. How accurate was our training?
#

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

sns.set_context('talk')

def plotTraining(df_false, df_true):
    """
    Plots training data set to visualize admission/non-admission categories.
    """

    sns.regplot("exam1", "exam2", df_false, fit_reg=False,
                marker="x",
                scatter_kws={"color": "blue", "s": 50},
                label="Not Admitted")
    ax = plt.gca()
    sns.regplot("exam1", "exam2", df_true, fit_reg=False, ax=ax,
               scatter_kws={"color": "red", "s": 50}, label="Admitted")

    ax.legend(loc='best')
    plt.xlim([30,100])
    plt.ylim([30,100])

    return ax

def plotDecisionBoundary(model, train, ax):
    """
    Plots probability contours (and decision boundary) of logistic
    regression model.
    """

    data_x1 = train[:,0]
    data_x2 = train[:,1]
    x1_grid = np.linspace(np.min(data_x1)-0.5, np.max(data_x1)+0.5, 101)
    x2_grid = np.linspace(np.min(data_x2)-0.5, np.max(data_x2)+0.5, 101)
    xx, yy = np.meshgrid(x1_grid, x2_grid)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired,alpha=0.4)

    plt.xlim([30,100])
    plt.ylim([30,100])

    return

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--data', type=str, default='ex2data1.txt',
    help='''Data set of admissions scores to use in first part of exercise. ''')
parser.add_argument(
    '--plot',action='store_true', default=False,
    help='''Turn plotting features on.''')
parser.add_argument(
    '--reg',type=float, default=1.0,
    help='''Inverse of regularization strength.''')
args = parser.parse_args()

df = pd.read_csv(args.data, names=['exam1','exam2','admission'])
print df.head()

# Step 1: Visualize training data in scatter plot
if args.plot:
    print "\n  scores of 0: ",len(df[df['admission'] == 0].index)
    print "  scores of 1: ",len(df[df['admission'] == 1].index)
    ax = plotTraining(df[df['admission'] == 0], df[df['admission'] == 1])


# Step 2: Fit logistic regression model using sklearn
train_X = np.column_stack( (np.array(df['exam1']), np.array(df['exam2'])) )
output_y = np.array(df['admission'])

# (Inverse) Regularization parameter controls the shape by quite a
# lot.  I found that reg ~ 100 matches the non-regularized case in the
# problem set in octave quite well.
regr = LogisticRegression(C=args.reg)
model = regr.fit(train_X, output_y)


# Step 3: Preliminary model evaluations:
# Check accuracy of training set: (compared to 60 % if we admitted
# everyone...)
print "Training Accuracy: ", model.score(train_X,output_y)

# Make prediction to test in problem set 2 from course:
print "predict for (45, 85): ",model.predict([45,85])
probability = model.predict_proba([45,85])
print probability
print "Admission probability: ",np.sqrt(probability[0][0]**2 + probability[0][1]**2)


# Step 4: Plot decision boundary
if args.plot:
    plotDecisionBoundary(model, train_X, ax)
    plt.show()
