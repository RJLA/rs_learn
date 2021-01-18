import glob
import os
import joblib
import pandas as pd
import numpy as np
from scipy.stats import stats
from math import sqrt
from sklearn.metrics import mean_squared_error

def reuse_model_reg(X_test,
               y_test,
               wildcard_name,
               ws = os.getcwd(),
               save = True):
    
    misc_output_path = os.path.join(os.getcwd(), 
                      'output_rs_learn',
                      'misc')     
    
    
    if not os.path.exists(misc_output_path):
        os.makedirs(misc_output_path)
        
    prediction_list = []
    feature_list = []
    for tuned_model in glob.glob(os.path.join(ws,
                                        'output_rs_learn',
                                        'tuned_models',
                                        f'%s*.sav'%wildcard_name)):
       
        model_trained = joblib.load(tuned_model)
        model_name = os.path.basename(tuned_model)[:-4]
        prediction = model_trained.predict(X_test)

        prediction_list.append(prediction)
        feature_list.append(model_name)


        slope, intercept, r_value, p_value, std_err = stats.linregress(y_test,
                                                                   prediction)
        r2 = r_value ** 2
        rmse = sqrt(mean_squared_error(prediction, y_test)) 
        percent_err =  ((prediction - y_test) / y_test) * 100
        mnb = np.mean(percent_err)
        
            
        print(f'{model_name} r: %.2f, r2: %.2f, rmse: %.2f, mnb: %.2f'%(r_value, r2, rmse, mnb))    


    df_prediction = pd.DataFrame(prediction_list).T
    df_prediction.columns = feature_list
    
    return df_prediction 
