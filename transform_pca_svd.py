from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
import pandas as pd
import os

def transform_pca_svd(df,
                file_name,
                n_pca = 1,
                n_svd = 1):

    misc_output_path = os.path.join(os.getcwd(), 
                      'output_rs_learn',
                      'misc')     

    if not os.path.exists(misc_output_path):
        os.makedirs(misc_output_path)

    df_norm = MinMaxScaler().fit_transform(df)
    
    pca_values = PCA(n_components = n_pca).fit_transform(df_norm)
    svd_values = TruncatedSVD(n_components = n_svd - 1).fit_transform(df_norm)
    
    df_pca = pd.DataFrame(pca_values,
                columns = ['PC%s'%i for i in range(1, n_pca + 1)]).set_index(df.index.values)

    df_pca.to_csv(os.path.join(misc_output_path,
    'pca_%s.csv'%file_name),
                index = False)
                
    df_svd = pd.DataFrame(svd_values,
                columns = ['SVD%s'%i for i in range(1, n_svd)]).set_index(df.index.values)
    
    df_svd.to_csv(os.path.join(misc_output_path,
     'svd_%s.csv'%file_name),
            index = False)

    print('''
    Done transforming data into principal components
    and single values.''')

    return df_pca, df_svd