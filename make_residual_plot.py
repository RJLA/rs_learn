from scipy.stats import skew, kurtosis, norm, iqr, stats
import matplotlib.pyplot as plt
import numpy as np
import os 
from math import sqrt
from sklearn.metrics import mean_squared_error

def make_residual_plot(actual,
                  predicted, 
                  bins,
                  file_name,
                  xlabel,
                  labelsize = 15,
                  titlesize= 15,
                  legendsize= 15,
                  ticksize = 15,
                  figsize = (8,8)):
                  
    graph_output_path = os.path.join(os.getcwd(), 
                        'output_rs_learn',
                        'graphs')
        
    if not os.path.exists(graph_output_path):
        os.makedirs(graph_output_path)                  
    
    f = plt.figure(figsize = figsize)
    
    #compute metrics
    error = predicted - actual
    error_ave = error.mean()
    error_std = error.std()
    error_min = error.min()
    error_max = error.max()
    error_iqr = iqr(error)
    error_q1 = np.quantile(error, 0.25)
    error_q3 = np.quantile(error, 0.75)
    ll = error_q1 - (1.5 * error_iqr)
    ul = error_q3 + (1.5 * error_iqr)    
    rmse = sqrt(mean_squared_error(actual, predicted))   
    dist_error_bins = bins    
#    predicted_ave = predicted.mean()

    
    #scatter 
    grid = plt.GridSpec(8, 8, 
                        hspace = 0.2, 
                        wspace = 0.05)
    
    main_ax = f.add_subplot(grid[:-1, 1:])
    y_hist = f.add_subplot(grid[:-1, 0], 
                           sharey = main_ax)  
    
    #make zero line 
    main_ax.axhline(0, 
            color = 'r', 
            linestyle = '-', 
            linewidth = 1,
            label = 'zero line')    
    
    #make outlier line
    main_ax.axhline(ll, 
            color = 'r', 
            linewidth = 1,
            linestyle = '--',                    
            label = '1.5 IQR')
    
    main_ax.axhline(ul, 
            color = 'r', 
            linestyle = '--',                    
            linewidth = 1)
    
    #make +/- rmse line
    main_ax.axhline(-rmse, 
            color = 'b', 
            linewidth = 1,
            linestyle = ':',                    
            label = 'rmse: +/- %.2f'%rmse)

    main_ax.axhline(rmse, 
            color = 'b', 
            linewidth = 1,
            linestyle = ':')

    #plot data
    main_ax.scatter(predicted, 
               error,
                marker = 's',
               color = 'k',
               facecolors = 'None',
               linewidth = 1,
               s = 25)
#    main_ax.set_xticks(fontsize = ticksize)
    #hide y axis
    main_ax.axes.get_yaxis().set_visible(False)
    
    #show ave of predicted
#    j = main_ax.scatter(predicted_ave, 
#           0,
#           color = 'green',
#           alpha = 1,
#          marker = 'o',
#          linewidth = 2)
#
#    j.set_zorder(20)
      
#    txt ='   Predicted_μ: %.2f'%predicted_ave 
#              
#    props = dict(boxstyle = 'round', 
#                 facecolor = 'w', 
#                 alpha = 1)
    
#    main_ax.text(predicted_ave, 
#             0, 
#             txt, 
#             fontsize = 8,
#             bbox = props)
    
    #put text
    txt = 'ε_μ: %.2f,ε_σ: %.2f \nε_min: %.2f, ε_max: %.2f'%(error_ave,
                                        error_std,
                                        error_min,
                                        error_max)  
                                             
    # #make psuedo element to add to legend
    main_ax.axvline(0, 
                alpha = 0,
                label = txt)    

    main_ax.legend(loc = 'best',
              framealpha = 0.9,
              fontsize = legendsize,
              facecolor = 'grey')

    #distribution error
    n, bins, patches = y_hist.hist(error, 
                                bins = dist_error_bins, 
                                density = 1, 
                                facecolor = 'white', 
                                edgecolor = 'k', 
                                linewidth = 1,
                                   orientation='horizontal')

    y_hist.invert_xaxis()
    
    y_hist.axhline(0, 
                   color = 'r', 
                   linestyle = '-', 
                   linewidth = 1)
    
    y_hist.axes.get_xaxis().set_visible(False)
    
    #put xy labels, title, and legend
    main_ax.set_xlabel(xlabel, 
               fontsize = labelsize)
    main_ax.tick_params(labelsize = ticksize)
#    main_ax.title
    plt.xticks(fontsize = ticksize)
    plt.yticks(fontsize = ticksize)


#    m = plt.legend(loc = 'best',
#              framealpha = 0.9,
#              fontsize = legendsize,
#              facecolor = 'grey')
              
#    m.set_zorder(21)
    
    y_hist.set_ylabel('Errors', 
               fontsize = labelsize)
    
    main_ax.set_title('Error distribution and residual plot', 
                      loc = 'left',
              fontsize = titlesize)
    
    main_ax.set_xlim(predicted.min(),
                 predicted.max())
    f.tight_layout()
    #save figure
    plt.savefig(os.path.join(graph_output_path,
    '%s_residuals.png'%file_name),
               dpi = 300)
    
    plt.show()


# import pandas as pd
# df = pd.read_csv("sample_dataframe.csv")
# make_residual_plot(df.iloc[:,0], 
                 # df.iloc[:,1],
                  # bins = 50,
                  # file_name = 'chla-results2',
                  # xlabel = 'Chlorophyll (u/L)',
                       # labelsize = 18,
                       # titlesize= 18,
                       # legendsize= 20,
                       # ticksize = 20,
                      # figsize = (10,10))




#make_scatter(df.iloc[:,0], 
#                 df.iloc[:,1],
#                 file_name = 'test',
#                 xlabel = 'Prediction',
#                 ylabel = 'Actual',
#                 title = 'Actual vs. Predicted',
#                 labelsize = 15,
#                 same_xy = [0,100],
##                     show_ave = True,
#                     rmse = True,
##                     show_ave = False,
##                     rmse = False
#                     )

