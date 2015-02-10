"""
Uplift-Modeling with Logistic Regression
"""
# Author: Michael Fitzke <m.fitzke@gmail.com>

from sklearn import linear_model
import numpy as np

class LogisticUplift(linear_model.LogisticRegression):
 

  def fit(self,X,T,y):
    #setting up Treatment Variables X*T
    X_T = X*T
    X = np.concatenate((X,X_T), axis=1)
    X = np.concatenate((X,T), axis = 1)
    super(LogisticUplift, self).fit(X, y)


   def predict_uplift(self, X,T)
   
    X_T = np.concatenate((X,X,T), axis=1)  
    X_no = np.concatenate((X,np.zeros_like(X),np.zeros_like(T)), axis=1)
    	
    y_T = super(LogisticUplift, self).predict_proba(X_T)
    y_no = super(LogisticUplift, self).predict_proba(X_no)
    
    return y_T - y_no
