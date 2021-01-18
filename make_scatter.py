import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import stats
from math import sqrt
from sklearn.metrics import mean_squared_error

def make_scatter(X, 
                 y,
                 file_name,
                 xlabel,
                 ylabel,
                 title,
                 same_xy = False,
                 show_stat = True,
                 labelsize = 15,
                 legendsize = 15,
                 titlesize = 15,
                 ticksize = 15):
 
    graph_output_path = os.path.join(os.getcwd(), 
                      'output_rs_learn',
                      'graphs')

    if not os.path.exists(graph_output_path):
        os.makedirs(graph_output_path)

    print()
    print('''
    args: X, y, unit, file_name, xlabel, ylabel, title, 
    show_ave = True; if true average for x and y will be shown
    rmse = True; if true rmse will be computed
          ''')  
            
    
    f, ax = plt.subplots(figsize = (8,8))
    
    # get x limits and y limits
    x_min = X.min()
    x_max = X.max()
    y_min = y.min()
    y_max = y.max()

    ax.set(xlim = (x_min, x_max), 
           ylim = (y_min, y_max))    
    
    #plot the regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
    
    p = intercept + slope * X
    
    ax.plot(X, 
            p, 
            '-',
            linewidth = 1,
            color = 'red',
            label = 'Line of best fit')    
    
    #plot data
    ax.scatter(X, 
               y,
               color = 'k',
               alpha = 1,
               facecolors = 'None',
               s = 25,
              marker = 's',
              linewidth = 1)
    
    if show_stat is True:
        



        #plot diagonal line
        if same_xy is False:
            diag_line,  = ax.plot(ax.get_xlim(), 
                         ax.get_ylim(), 
                         ls = '--', 
                         c = 'b',
                         label = 'x = y',
                         linewidth = 1,
                         alpha = 1)
        else:
            xy_axis = same_xy
            ax.set_xlim(xy_axis)
            ax.set_ylim(xy_axis)
            diag_line,  = ax.plot(ax.get_xlim(), 
                 ax.get_ylim(), 
                 ls = '--', 
                 c = 'b',
                 label = 'x = y',
                 linewidth = 1,
                 alpha = 1)
    
        #show average as point
        x_ave = X.mean()
        y_ave = y.mean()
        
        j = ax.scatter(x_ave, 
                   y_ave,
                   color = 'green',
                   alpha = 1,
                  marker = 'o',
                  linewidth = 8)
                  
        j.set_zorder(20)
        
        #compute accuracy metrics    
        percent_err =  ((X - y) / y) * 100
        rmse = sqrt(mean_squared_error(X, y)) 
        nrmse = np.std(percent_err)
        mnb = np.mean(percent_err)
        nmae = np.mean(np.abs(percent_err))
        r2 = r_value ** 2
        
        #put text    
        
        txt = f'r: %.2f, r2: %.2f\nrmse: %.2f\nnrmse: %.2f\nmnb: %.2f\nnmae: %.2f\nx̅ x&y: %.2f, %.2f\ny = %.2fx + %.2f'%(r_value,
                                                   r2, 
                                                   rmse,
                                                   nrmse,
                                                   mnb,
                                                   nmae,
                                                   x_ave,
                                                   y_ave,
                                                   slope,
                                                   intercept)          
        
    else:
        r2 = r_value ** 2
        
        #put text
        txt ='r: %.2f, r2: %.2f\ny = %.2f0x + %.2f'%(r_value,
                                                   r2, 
                                                   slope,
                                                   intercept)     
                                   
    #make psuedo point to add to legend
    ax.scatter(x_min,
              y_min,
              label = txt,
              alpha = 0)
                                
    #put xy labels, title, and legend
    plt.xlabel(xlabel, 
              fontsize = labelsize)
    
    plt.ylabel(ylabel, 
              fontsize = labelsize)
    
    plt.xticks(fontsize = ticksize)
    plt.yticks(fontsize = ticksize)

    plt.title(title,
              loc = 'left',
              fontsize = titlesize)
    
    m = plt.legend(loc = 'best',
              framealpha = 0.9,
              fontsize = legendsize,
              facecolor = 'grey')
              
    m.set_zorder(21)

    
    #save figure
    plt.tight_layout()

    plt.savefig(os.path.join(graph_output_path,
                '%s_scatter.png'%file_name),
               dpi = 300)
    
    plt.show()
    
#import pandas as pd
#df = pd.read_csv("sample_dataframe.csv")
#
#make_scatter(df.iloc[:,0], 
#                 df.iloc[:,1],
#                 file_name = 'test',
#                 xlabel = 'Prediction',
#                 ylabel = 'Actual',
#                 title = 'Actual vs. Predicted',
#                 labelsize = 15,
#                 same_xy = [0,100],
#                 show_stat = False
#                     )

