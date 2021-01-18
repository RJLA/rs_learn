import matplotlib.pyplot as plt
from scipy.stats import anderson
from statsmodels.graphics.gofplots import qqplot
import os

def make_qqplot(X,
               file_name,
                  labelsize = 15,
                  titlesize= 15,
                  legendsize= 15,
                  ticksize = 15):
    
    graph_output_path = os.path.join(os.getcwd(), 
                    'output_rs_learn',
                    'graphs')
        
    if not os.path.exists(graph_output_path):
        os.makedirs(graph_output_path)   
    
    f, ax = plt.subplots(figsize = (8,8))

    result = anderson(X)
    
    test_stat = result.statistic
    crit_val = result.significance_level[2]
    
    if crit_val < result.statistic:
        decision = 'Reject the null hypothesis.\nData does not follow a normal distribution'
    else:
        decision = 'Fail to reject the null hypothesis.\nData follows a normal distribution'
    
    # plot qq
    txt = 'Anderson Darling statistics: %.2f\ncritical value (α=0.05): %.2f\n%s'%(test_stat,
                                                                            crit_val,
                                                                                  decision)
    
    qqplot(X, 
           ax = ax,
           line='s',
           color = 'k')
    
    plt.xlabel('Theoretical quantiles',
              fontsize = labelsize)
    
    plt.ylabel('Sample quantiles',
              fontsize = labelsize)   
    
    #plot psuedo line for txt legend
    plt.axvline(0, 
            alpha = 0,
            label = txt)    

    plt.legend(loc = 'best',
              framealpha = 0.9,
              fontsize = legendsize,
              facecolor = 'grey')
    
    plt.title('Quantile-Quantile plot',
             fontsize = titlesize,
             loc = 'left')
    
    plt.xticks(fontsize = ticksize)
    plt.yticks(fontsize = ticksize)    
    
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(graph_output_path,
    '%s_qqplot.png'%file_name),
               dpi = 300)
    
    plt.show()

#import pandas as pd
#df = pd.read_csv("sample_dataframe.csv")
#err = df.iloc[:,0] - df.iloc[:,1]
#make_qqplot(err,
#               file_name = 'chla-err2',
#               legendsize = 15)