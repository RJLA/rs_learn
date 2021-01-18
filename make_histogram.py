import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import anderson
import matplotlib.pyplot as plt
import os
from scipy.stats import skew, kurtosis, norm, iqr, stats
from math import sqrt
from sklearn.metrics import mean_squared_error

def make_histogram(X,
              num_bins,
              file_name,
              title,
              xlabel,
                  labelsize = 15,
                  titlesize= 15,
                  legendsize= 15,
                  ticksize = 15,
              x_axis = False,
              show_stat = False):

    graph_output_path = os.path.join(os.getcwd(), 
                      'output_rs_learn',
                      'graphs')

    if not os.path.exists(graph_output_path):
        os.makedirs(graph_output_path)

    print()
    print('''
    args: X, num_bins, 
    file_name, title, xlabel
          ''')  
    
    f, ax = plt.subplots(figsize = (8,8))

    n, bins, patches = plt.hist(X, 
                                num_bins, 
                                density = 1, 
                                facecolor = 'white', 
                                edgecolor = 'k', 
                                linewidth = 1)
    
    x_mean = X.mean()
    x_std = X.std()
    x_min = X.min()
    x_max = X.max()
    skewness = skew(X)
    kurt = kurtosis(X)

    result = anderson(X)
    
    test_stat = result.statistic
    crit_val = result.significance_level[2]
    
    if crit_val < result.statistic:
        decision = 'Reject Ho.\nNon normal'
    else:
        decision = 'Fail to reject Ho.\nNormal'
    
    txt_ad = 'AD stat: %.2f\ncv (α=0.05): %.2f\n%s'%(test_stat,
                                crit_val,
                                decision)
    

    # add a pdf line
    y = norm.pdf(bins,
                 x_mean, 
                 x_std)

    ax.plot(bins, 
            y, 
            'k-',
            linewidth = 1,
            label = 'Normal pdf')
    
    #plot min, mean, max, std
    plt.axvline(x_mean, 
                color = 'k', 
                linestyle = '--', 
                linewidth = 2,
                label = 'mean')

    plt.axvline(x_mean + (2 * x_std), 
                color = 'grey', 
                linestyle = '--', 
                linewidth = 1,
                label = '+/- 2σ/3σ')

    plt.axvline(x_mean - (2 * x_std), 
                color = 'grey', 
                linestyle = '--', 
                linewidth = 1)                

    plt.axvline(x_mean + (3 * x_std), 
                color = 'grey', 
                linestyle = '--', 
                linewidth = 1)

    plt.axvline(x_mean - (3 * x_std), 
                color = 'grey', 
                linestyle = '--', 
                linewidth = 1)
    if x_axis is False:
        plt.xlim(x_min, x_max)
    else:
        plt.xlim(x_axis)
    
                
    #put xy labels, title, and legend
    plt.xlabel(xlabel,
              fontsize = labelsize)
    
    plt.ylabel('Probability', 
              fontsize = labelsize)
    
    plt.title(title,
              loc = 'left',
             fontsize = titlesize)
                            
    
    #put text
    txt ='μ = %.2f\nσ = %.2f\nmin: %.2f\nmax: %.2f\nskewness: %.2f\nkurtosis: %.2f\n'%(x_mean,
                                             x_std,
                                             x_min,
                                             x_max,
                                             skewness,
                                             kurt)  
                                             
    if show_stat is False:
        pass
    else:

        #make psuedo element to add to legend

        plt.axvline(x_mean, 
                alpha = 0,
                label = txt)   
    
        #plot psuedo line for txt legend
        plt.axvline(0, 
                alpha = 0,
                label = txt_ad)  
    
    plt.xticks(fontsize = ticksize)
    plt.yticks(fontsize = ticksize)

    plt.legend(loc = 'best',
              framealpha = 0.9,
              fontsize = legendsize,
              facecolor = 'grey')
              

    #save figure
    plt.tight_layout()
    plt.savefig(os.path.join(graph_output_path,
    '%s_histogram.png'%file_name),
               dpi = 300)
    
    plt.show()
    plt.close()