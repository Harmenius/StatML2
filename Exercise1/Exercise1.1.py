__author__ = 'Harmen'

import numpy as np


x_b = np.array([0,0])

mu = np.array([1, 0, 1, 2]).T
Sigma_tilde = np.array([ [ 0.14,-0.30, 0.00, 0.20]
                        ,[-0.30, 1.16, 0.20,-0.80]
                        ,[ 0.00, 0.20, 1.00, 1.00]
                        ,[ 0.20,-0.80, 1.00, 2.00]])
Lambda_tilde = np.linalg.inv(Sigma_tilde)
Lambda = {'aa':Lambda_tilde[0:2,0:2], 'ab':Lambda_tilde[0:2,2:4],
          'ba':Lambda_tilde[2:4,0:2], 'bb':Lambda_tilde[2:4,2:4]}
Sigma_p = np.linalg.inv(Lambda_tilde[0:2,0:2])
mu_p = Sigma_p.dot(Lambda['aa'].dot(mu[:2]) - Lambda['ab'].dot(x_b - mu[2:]))

print mu_p