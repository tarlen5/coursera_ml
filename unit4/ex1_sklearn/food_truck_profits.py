#! /usr/bin/env python
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# Use a single variable simple linear model to predict profits for
# opening a food truck in a given city (with no regularization or
# cross validation).
#
# This follows along with the Coursera ML course assignment 1.
#


def getData(datafile,plot=True):
    """
    Returns datafile (a csv file) as a DataFrame object and makes a
    scatter plot with seaborn lmplot
    """
    frame = pd.read_csv(datafile,names=['population','profit'])

    if plot:
        # Seaborn has its own version of a linear regression fit line:
        sns.lmplot("population","profit",frame,size=7)

        plt.xlabel('city population (in 10,000s)')
        plt.ylabel('Profit (in $10,000s)')

        print("\n  PAUSED...Close figure to continue...")
        plt.show()

    return frame


from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn import linear_model

sns.set_style('darkgrid')
sns.set_context('talk')


datafile = 'ex1data1.txt'

# Step 1: plot as scatter plot, and return dataframe
frame = getData(datafile,plot=False)

# LinearRegression takes a feature matrix of shape (# train examples, # features)
# and an output vector of a single column (same # of train ex.)
population = np.array(frame['population'])
population = population.reshape(len(population),1)
profit = np.array(frame['profit'])

# Step 2: Fit linear regression model to dataset
regr = linear_model.LinearRegression()
regr.fit(population, profit)

# Print coefficients:
print("Coefficiencts: \n", regr.coef_)


# Step 3: Predict for profits for these three new populations: 35,000,
# 70,000, and 150,000
new_pop = np.array([3.5, 7.0, 15.0])
new_pop = new_pop.reshape(len(new_pop), 1)
predictions = regr.predict(new_pop)
for ii,prediction in enumerate(predictions):
    print("  Population: %d, profits: %.1f"%(int(new_pop[ii]*10000),
                                             prediction*10000))

# Finally, plot fit to the population:
plt.scatter(population,profit)
plt.plot(population, regr.predict(population),lw=3)
plt.title("Linear Regression fit to Food Truck Profit vs. City Population")
plt.xlabel('Population (x 10,000)')
plt.ylabel('Profits (x $10,000)')
plt.show()
