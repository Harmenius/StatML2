__author__ = 'Harmen'

import numpy as np
name = {}

mu_tilde = np.array([1, 0, 1, 2]).T
Sigma_tilde = np.array([ [ 0.14,-0.30, 0.00, 0.20]
                        ,[-0.30, 1.16, 0.20,-0.80]
                        ,[ 0.00, 0.20, 1.00, 1.00]
                        ,[ 0.20,-0.80, 1.00, 2.00]])
Lambda_tilde = np.linalg.inv(Sigma_tilde)
Lambda_aa = np.linalg.inv(Lambda_tilde[0:2,0:2])
print Lambda_aa