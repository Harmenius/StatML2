__author__ = 'Harmen'

import numpy as np
import pandas
import matplotlib.pyplot as plot
import seaborn

# Exercise 1.1.1
# Setting up the given values
x_b = np.array([0,0])
mu = np.array([1, 0, 1, 2]).T
Sigma_tilde = np.array([ [ 0.14,-0.30, 0.00, 0.20]
                        ,[-0.30, 1.16, 0.20,-0.80]
                        ,[ 0.00, 0.20, 1.00, 1.00]
                        ,[ 0.20,-0.80, 1.00, 2.00]])

# Calculating conditional precision matrix
Lambda_tilde = np.linalg.inv(Sigma_tilde)
Lambda = {'aa':Lambda_tilde[0:2,0:2], 'ab':Lambda_tilde[0:2,2:4],
          'ba':Lambda_tilde[2:4,0:2], 'bb':Lambda_tilde[2:4,2:4]}

# Calculating conditional variances and means
Sigma_p = np.linalg.inv(Lambda_tilde[0:2,0:2])
mu_p = Sigma_p.dot(Lambda['aa'].dot(mu[:2]) - Lambda['ab'].dot(x_b - mu[2:]))

# Exercise 1.1.2
# Sample from P_a|b
mu_t = np.random.multivariate_normal(mu_p, Sigma_p)

# Exercise 1.1.3
# Building a 3D plot of P_a|b
P_dist = np.random.multivariate_normal(mean=mu_p, cov=Sigma_p, size=1000)
pandas.DataFrame(P_dist, columns=["X", "Y"])
P_plot = seaborn.kdeplot(P_dist, shade=True)
# plot.show()

# Exercise 1.2.1
# Define variance
Sigma_t = np.array([[2.0, 0.8], [0.8, 4.0]])

# Draw samples
x = np.random.multivariate_normal(mu_t, Sigma_t, size=1000)
np.savetxt("sample1.txt", x)

# Compute Maximum Likelihood
mu_ML = np.mean(x,0)
Sigma_ML = np.mean(np.array([[x[:,0]*x[:,0], x[:,0]*x[:,1]], [x[:,1]*x[:,0], x[:,1]*x[:,1]]]).T, 0)