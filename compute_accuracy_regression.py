from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import stats
import os
def compute_accuracy_regression(actual,
                    predicted):
    
    
    misc_output_path = os.path.join(os.getcwd(), 
                      'output_rs_learn',
                      'misc')     

    if not os.path.exists(misc_output_path):
        os.makedirs(misc_output_path)
        
    r2 = r2_score(actual, predicted)
    
    rmse = sqrt(mean_squared_error(actual,
                                   predicted))
    
        
    print('r2: %.2f, rmse: %.2f'%(r2, rmse))
    return r2, rmse