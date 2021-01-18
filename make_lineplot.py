import matplotlib.pyplot as plt
import os

def make_lineplot(x,
                 y,
                 file_name,
                 xlabel,
                 ylabel,
                 title,
                 labelsize = 15,
                 titlesize = 15,
                 threshold = None,
                 two_std = None):
    
    graph_output_path = os.path.join(os.getcwd(), 
                      'output_rs_learn',
                      'graphs')

    if not os.path.exists(graph_output_path):
        os.makedirs(graph_output_path)

    print()
    
    f, ax = plt.subplots(figsize = (8, 8))


    ax.plot(x, 
            y, 
            '-',
            linewidth = 1,
            color = 'grey') 
    
    
    plt.xlabel(xlabel, 
               fontsize = labelsize)
    
    plt.ylabel(ylabel, 
               fontsize = labelsize)
    
    plt.title(title, 
              fontsize = titlesize,
              loc = 'left')
    
    
    if threshold is None:
        pass
    else:   
        plt.axhline(threshold, 
                color='k',
                label = 'threshold')
        
        
    if two_std is None:
        pass
    else:   
        plt.axhline(two_std[0], 
                color='red',
                ls = '--',
                linewidth = 0.8)
        
        plt.axhline(two_std[1], 
                color='red',
                ls = '--',
                linewidth = 0.8)


    plt.tight_layout()

    plt.savefig(os.path.join(graph_output_path,
                f'{file_name}_line.png'),
               dpi = 300)
    
    plt.show()

#
#import pandas as pd
#
#df = pd.read_csv('sample_dataframe.csv')
#make_lineplot(df['x1'],
#              df['x2'],
#             file_name = 'test',
#             xlabel = 'pred',
#             ylabel = 'values',
#             title = 'bp',
#             threshold = 20,
#             two_std = [21,22])

#
#def make_boxplot(data,
#                 file_name,
#                 xlabel,
#                 ylabel,
#                 title):