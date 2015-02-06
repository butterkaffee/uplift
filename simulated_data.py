import numpy as np
from math import exp

def simulate_linear_factors():
  num_samples = 300  

  # The desired mean values of the sample.
  mu = np.array([45, 935, 680.5, 654.5])

  # The desired covariance matrix.
  r = np.array([
        [  13*13, 507, 152, 355],
        [ 507,  150*150,  6.75, 15.75],
        [ 152,  6.75,  150*150, 4.725],
        [355, 15.75, 4.725, 70*70]
    ])

  # Generate the random samples.
  return np.random.multivariate_normal(mu, r, size=num_samples)

def simulate_logistic(X,v,beta0):
  y = []
  for r in X: 
    if exp(beta0+np.dot(r[0:3],v))/(1+exp(beta0+np.dot(r[0:3],v))) < 0.5:
      y.append(0)
    else: 
      y.append(1)

  return np.array([y])

def simulate():
  X_T = simulate_linear_factors()
  v_T = np.array([0.02, 0.005, 0.00155])
  beta0_T = -7.5
  y_T = simulate_logistic(X_T,v_T, beta0_T)
  T_T = np.array([[1]*len(X_T)])

  X_C = simulate_linear_factors()
  v_C = np.array([0.04, 0.005, 0.00165])
  beta0_C = -8
  y_C = simulate_logistic(X_C, v_C, beta0_C)
  T_C = np.array([[0]*len(X_C)])

  return (np.concatenate((X_T, X_C)), np.concatenate((T_T.T, T_C.T)), np.concatenate((y_T.T, y_C.T))) 



