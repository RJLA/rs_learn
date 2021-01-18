import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
def reducer_umap(df,
              file_name,
              n_neighbors = 15, 
              min_dist = 0.1, 
              n_components = 2, 
              metric = 'euclidean'):
    
    misc_output_path = os.path.join(os.getcwd(), 
                      'output_rs_learn',
                      'misc')     

    if not os.path.exists(misc_output_path):
        os.makedirs(misc_output_path)

    norm = MinMaxScaler().fit_transform(df)
    
    fit = umap.UMAP(n_neighbors = n_neighbors,
        min_dist = min_dist,
        n_components = n_components,
        metric = metric)
    u = fit.fit_transform(norm)
    
    df_umap = pd.DataFrame(u, 
                           columns = ['UMAP_%s'%i \
                                      for i in range(1, 
                                    n_components + 1)]).set_index(df.index.values)
    df_umap.to_csv(os.path.join(misc_output_path,
    'umap_%s.csv'%file_name),
                index = False)
    
    return df_umap
