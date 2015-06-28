# coursera_ml
Problem sets and assignments for the coursera machine learning course taught by Stanford Prof. Andrew Ng, completed in **octave/matlab**. Secondly, these problems have been expanded on for my own usage and development purposes in **python using the sckikit-learn library**.

### Unit 4: *Project 1-Linear Regression*
* Implemented a Linear Regression model to make predictions in several scenarios:
  * with one variable to predict profits for a food truck
  * with multiple variables to predict housing prices in Portland, Oregon.
* Both were implemented in `octave/matlab` for the course work and in `Python using sklearn`
* Files completed:
  * in ex1/
    * `warmUpExercise.m`, `plotData.m`, `gradientDescent.m`, `computeCost.m`
  * ex1_sklearn/
    * `food_truck_profits.py`, `housing_prices.py`

### Unit 6: *Project 2-Logistic Regression*
1. To model student acceptance rates at a university (linear model/decision boundary)
2. To model whether microchips from a fabrication plant passes quality assurance (non-linear model/decision boundary) (note: Not completed in sklearn, only in octave/matlab.)

* Projects completed in `octave/matlab` for the course work and in `Python using sklearn`
* files completed 
  * in ex2/: `plotData.m`, `sigmoid.m`, `costFunction.m`, `predict.m`, `costFunctionReg.m`
  * in ex2_sklearn/: `university_admissions.py`

### Unit 8: *Project 3-Image Processing with Multi-class Classification and Neural Networks*
* Implementations: One-vs-all logistic regression and neural networks to identify sample of hand-written digit examples.
  * One-vs-all logistic regression implemente in octave and python with sklearn.
  * Neural network feedforward propagation was performed on a model that already had the network parameters trained for us.
* Training set: 5000 examples of handwritten digits, which is a subset of the MNIST handwritten digit dataset (http://yann.lecun.com/exdb/mnist/).
* Files completed:
  * in ex3/: `lrCostFunction.m`, `oneVsAll.m`, `predictOneVsAll.m`, `predict.m`
  * in ex3_sklearn/: `lrDigitRecog.py`
   * TO DO: `nnDigitRecog.py` (figure out which python framework to do neural network classification. Probably the `Pylearn2` library).

### Unit 9: Assignment 4
* Neural Networks Learning

### Unit 10: Assignment 5
* Regularized Linear Regression Bias vs. Variance

### Unit 12: Assignment 6
* Support Vector Machines

### Unit 14: Assignment 7
* Image Processing with K-means Clustering and Principal Component Analysis

### Unit 16: Assignment 8
* Anomaly Detection and Recommender Systems for Movie recommendations
