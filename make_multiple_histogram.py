import os
import matplotlib.pyplot as plt
def make_multiple_histogram(X,
                      bins,
                      file_name,
                      color_list,
                      xlabel,
                  labelsize = 15,
                  titlesize= 15,
                  legendsize= 15,
                  ticksize = 15,
                  alpha = 0.5):

    graph_output_path = os.path.join(os.getcwd(), 
                      'output_rs_learn',
                      'graphs')

    if not os.path.exists(graph_output_path):
        os.makedirs(graph_output_path)
    
    print('''
    args: X,
    bins,
    file_name,
    color_list
         ''')

    print('''
    more color codes in this site
    https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    ''')
    print()

    f, ax = plt.subplots(figsize = (8,8))
    
    kwargs = dict(alpha = alpha, 
                  bins = bins, 
                  density = True, 
                  stacked = True)
    
    for i,j in zip(X.columns,
                   color_list):
        
        ax.hist(X[i],
                **kwargs,
                label = '%s'%i,
               color = j)
    
    plt.xlabel(xlabel,
             fontsize = labelsize)
    
    plt.ylabel('Probability',
             fontsize = labelsize)

    plt.title('Probability Distribution',
                loc = 'left',
                fontsize = titlesize)
    
    plt.legend(loc = 'best',
              framealpha = 1,
              fontsize = legendsize)
    
    plt.xticks(fontsize = ticksize)
    plt.yticks(fontsize = ticksize)    
    plt.tight_layout()
              
    plt.savefig(os.path.join(graph_output_path,
    'stacked_histogram_%s.png'%file_name),
               dpi = 300)
    plt.show()
    
#    
#import pandas as pd
#df = pd.read_csv("sample_dataframe.csv")
#make_multiple_histogram(df.iloc[:,:2],
#                      bins = 50,
#                      file_name = 'results2',
#                      color_list = ['r','grey'],
#                            xlabel = 'Chlorophyll-a (u/L)',
#                      alpha = 0.5,
#                            labelsize = 18,
#                            titlesize= 18,
#                            legendsize= 15,
#                            ticksize = 20)