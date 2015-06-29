# coursera_ml
Programming projects for the Coursera Machine Learning course taught by Stanford Prof. Andrew Ng, completed in **octave/matlab**. I have also implemented some of these projects in **python using the scikit-learn library** for my own research and development purposes.

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
2. To model whether microchips from a fabrication plant pass quality assurance (non-linear model/decision boundary) (note: Not completed in sklearn, only in octave/matlab.)

* files completed:
  * in ex2/: `plotData.m`, `sigmoid.m`, `costFunction.m`, `predict.m`, `costFunctionReg.m`
  * in ex2_sklearn/: `university_admissions.py`

### Unit 8: *Project 3-Image Processing with Multi-class Classification and Neural Networks*
* Implementations: One-vs-all logistic regression and neural networks to identify sample of hand-written digit examples.
  * One-vs-all logistic regression implemented in octave and python with sklearn.
  * Neural network feedforward propagation was performed on a model that already had the network parameters trained for us.
* Training set: 5000 examples of handwritten digits, which is a subset of the MNIST handwritten digit dataset (http://yann.lecun.com/exdb/mnist/).
* Files completed:
  * in ex3/: `lrCostFunction.m`, `oneVsAll.m`, `predictOneVsAll.m`, `predict.m`
  * in ex3_sklearn/: `lrDigitRecog.py`
   * TO DO: `nnDigitRecog.py` (figure out which python framework to do neural network classification. Probably the `Pylearn2` library).

### Unit 9: *Project 4-Image Classification with Neural Networks*
* Dataset and learned hidden parameters (`Theta1` and `Theta2`) are same as in *Project 3* 
* In this project was implemented
  * regularized cost function
  * backpropagation algorithm to compute gradient of nn cost function
  * A number of additional features were implemented as well, including randomizing the initial weights and checking the gradient
  * Files completed (octave/matlab only):
   * ex4/: `sigmoidGradient.m`, `randInitializeWeights.m`, `nnCostFunction.m` (bulk of work here)
* TO DO: (To extend this assignment and implement in python) Perform neural network training and predictions, using `Pylearn2` or similar.

### Unit 10: *Project 5-Bias/Variance Studies with Regularized Linear Regression*
* Regularized linear regression to predict amount of water flowing into/out of a dam using change in water level in a reservoir.
* Diagnostics of debugging learning algorithms and effects of bias vs. variance.
* Dataset division into *training set*, *cross validation set*, and *test set* is introduced and implemented.
* Learning curves are generated
* Files completed:
 * ex5/: `linearRegCostFunction.m `, `learningCurve.m`, `polyFeatures.m`, `validationCurve.m`

### Unit 12: *Project 6-Support Vector Machines (SVM) for Spam Classification*
* 
* Files completed:
  * ex6/: `gaussianKernel.m`, `dataset3Params.m`, `processEmail.m`, `emailFeatures.m`
  * ex6_sklearn/: 

### Unit 14: Assignment 7
* Image Processing with K-means Clustering and Principal Component Analysis

### Unit 16: Assignment 8
* Anomaly Detection and Recommender Systems for Movie recommendations
