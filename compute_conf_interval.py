import numpy as np
from scipy import stats
import math

def compute_conf_interval(data, 
                interval = 0.95, 
                method = 't'):
    series = data    
    mean_val = series.mean()
    n = series.count()
    stdev = series.std()
    if method == 't':
        test_stat = stats.t.ppf((interval + 1)/2, n)
    elif method == 'z':
        test_stat = stats.norm.ppf((interval + 1)/2)
    
    lower_bound = mean_val - test_stat * stdev / math.sqrt(n)
    upper_bound = mean_val + test_stat * stdev / math.sqrt(n)
    
    print(f'Lower limit: {lower_bound}\n Upper limit: {upper_bound}')

#import pandas as pd
#
#df = pd.read_csv('sample_dataframe.csv')
#compute_conf_interval(df['x1'])