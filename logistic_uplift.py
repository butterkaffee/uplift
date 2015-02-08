"""
Uplift-Modeling with Logistic Regression
"""
# Author: Michael Fitzke <m.fitzke@gmail.com>

from sklearn import linear_model
import numpy as np

class LogisticUplift(linear_model.LogisticRegression):
  def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
               fit_intercept=True, intercept_scaling=1, class_weight=None,
               random_state=None, solver='liblinear', max_iter=100,
               multi_class='ovr', verbose=0):

    
    super(LogisticUplift,self).__init__(penalty, dual, tol, C, fit_intercept,
					 intercept_scaling, class_weight, random_state, solver,
					max_iter, multi_class, verbose)

  def fit(self,X,T,y):
    #setting up Treatment Variables X*T
    X_T = X*T
    X = np.concatenate((X,X_T), axis=1)
    X = np.concatenate((X,T), axis = 1)
    super(LogisticUplift, self).fit(X, y)


     
