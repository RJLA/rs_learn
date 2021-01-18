import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def make_heatmap(df,
                file_name,
                corr_mat = True,
                annot_size = 12,
                figsize = (8,8)):

    graph_output_path = os.path.join(os.getcwd(), 
                      'output_rs_learn',
                      'graphs')

    if not os.path.exists(graph_output_path):
        os.makedirs(graph_output_path)

    print()
    print('''
    args: dataframe, file_name,
    corr_mat = True; if true correlation matrix
    will be saved as csv format
          ''')    

    #get columns
    columns = df.columns

    #create correlation matrix
    corr_mat = np.corrcoef(df[columns].values.T)
    
    if corr_mat is True:
        corr_mat.to_csv(f'correlation_matrix_{file_name}.csv')
    else:
        pass
    
    plt.figure(figsize = figsize)
    
    hm = sns.heatmap(corr_mat,
                     cbar = True,
                     annot = True, 
                     square = False,
                     annot_kws = {'size': annot_size},
                     fmt='.1f',
                     yticklabels = columns,
                     xticklabels = columns,
                     cmap = "RdBu_r")
    
    hm.set_xticklabels(columns,
                       rotation = 90,
                      fontsize = 10)
                      
    hm.set_yticklabels(columns[:],
                       rotation = 360,
                      fontsize = 10)
                      
    plt.title('Correlation Heatmap of Features',
                loc = 'left',
                fontsize = 12)
                
    plt.tight_layout()
    
    plt.savefig(os.path.join(graph_output_path,
                        f'heatmap_{file_name}.png'),
                        dpi = 300)
                        
    plt.show()