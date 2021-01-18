from sklearn.decomposition import FastICA 
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
def transform_ica(df,
                  file_name,
                  n_ica,
                  max_iter = 200,
                  tol = 0.0001):
    
    
    misc_output_path = os.path.join(os.getcwd(), 
                      'output_rs_learn',
                      'misc')     

    if not os.path.exists(misc_output_path):
        os.makedirs(misc_output_path)
        
        
    ica = FastICA(n_components = n_ica, 
                  random_state = 0,
                  tol = tol,
                  max_iter = max_iter) 
    
    df_norm = MinMaxScaler().fit_transform(df)
    
    X_ica = ica.fit_transform(df_norm)
    
    df_ica = pd.DataFrame(X_ica,
            columns = [f'IC_{i}' \
                       for i in range(1, 
                                      n_ica + 1)]).set_index(df.index.values)

    df_ica.to_csv(os.path.join(misc_output_path,
     f'ica_{file_name}.csv'),
            index = False) 
    
    return df_ica 


