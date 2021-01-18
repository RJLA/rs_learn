from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os

def make_pca_svd_graph(df,
                file_name):
    
    graph_output_path = os.path.join(os.getcwd(), 
                    'output_rs_learn',
                    'graphs')
        
    if not os.path.exists(graph_output_path):
        os.makedirs(graph_output_path)   

    X_norm = MinMaxScaler().fit_transform(df)
    
    pca = PCA(n_components = df.shape[1])
    svd = TruncatedSVD(n_components = df.shape[1] - 1)

    pca.fit(X_norm)
    svd.fit(X_norm)
    
    tot_pca = sum(pca.explained_variance_)
    tot_svd = sum(svd.explained_variance_)
    
    variance_exp_pca = [(i / tot_pca) for i in sorted(pca.explained_variance_, 
                                        reverse = True)]
    variance_exp_svd = [(i / tot_svd) for i in sorted(svd.explained_variance_, 
                                    reverse = True)]
                                   
    cum_variance_exp_pca = np.cumsum(variance_exp_pca)
    cum_variance_exp_svd = np.cumsum(variance_exp_svd)
        
    x_axis = df.shape[1]

    def graph_pc(variance_exp,
                cum_variance_exp,
                xlabel,
                ylabel,
                title,
                name,
                range_n):
                
        f, ax = plt.subplots(figsize = (8, 8))
        
        plt.bar(range_n, 
                variance_exp,
                fill = False,
                label = 'Individual explained variance',
                color = 'k')
        
        plt.step(range_n, 
                 cum_variance_exp, 
                 where = 'mid',
                 label = 'Cumulative explained variance\n%s'%cum_variance_exp,
                 color = 'k')
           
        plt.ylabel(ylabel,
                fontsize = 10)
                
        plt.xlabel(xlabel,
                    fontsize = 10)
                    
        plt.title(title,
                    loc = 'left',
                    fontsize = 12)
                    
        plt.legend(loc = 'best',
                    framealpha = 1)
                    
        plt.tight_layout()

        plt.savefig(os.path.join(graph_output_path,
        '%s_%s.png'%(name,file_name)), 
                    dpi = 300)
        plt.show()
        plt.close()
                    
    graph_pc(variance_exp_pca,
                cum_variance_exp_pca,
                xlabel = 'Principal components',
                ylabel = 'Explained variance',
                title = 'Principal Component Analysis',
                name = 'pca',
                range_n = range(1, x_axis + 1))
    
    graph_pc(variance_exp_svd,
                cum_variance_exp_svd,
                xlabel = 'Singular value used',
                ylabel = 'Explained variance',
                title = 'Single value decomposition',
                name = 'svd',
                range_n = range(1,x_axis)) 