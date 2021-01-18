import numpy as np
from scipy.stats import chi2
from scipy.stats import norm


def compute_tolerance(data):
    n = len(data)
    dof = n - 1

    prop = 0.95
    prop_inv = (1.0 - prop) / 2.0
    gauss_critical = norm.isf(prop_inv)
    prob = 0.95
    chi_critical = chi2.isf(q=prob, df=dof)
    interval = np.sqrt((dof * (1 + (1/n)) * gauss_critical**2) / chi_critical)

    data_mean = np.mean(data)
    lower, upper = data_mean-interval, data_mean+interval
    print('%.2f to %.2f covers %d%% of data with a confidence of %d%%' % (lower, 
                                                                          upper, prop*100, prob*100))
#import pandas as pd
#
#df = pd.read_csv('sample_dataframe.csv')
#compute_tolerance(df['x1'])