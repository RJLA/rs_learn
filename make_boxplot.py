import matplotlib.pyplot as plt
import os

def make_boxplot(data,
                 file_name,
                 xlabel,
                 ylabel,
                 title,
                 labelsize = 15,
                 titlesize = 15,
                 rotation = 'vertical'):
        
    graph_output_path = os.path.join(os.getcwd(), 
                      'output_rs_learn',
                      'graphs')

    if not os.path.exists(graph_output_path):
        os.makedirs(graph_output_path)

    print()
    
    f, ax = plt.subplots(figsize = (8, 8))    
    
    bp = ax.boxplot(data.T)
    
    for box in bp['boxes']:
        box.set(color='k', 
                linewidth = 1)
        
    for median in bp['medians']:
        median.set(color='k', 
                   linewidth = 1)

    for whisker in bp['whiskers']:
        whisker.set(color='k', 
                    linewidth = 1)

    for cap in bp['caps']:
        cap.set(color='k', 
                linewidth = 1)

    for flier in bp['fliers']:
        flier.set(marker = 'o')

    ax.set_xticklabels(data.T.index,
                       fontsize = 10)

    plt.xlabel(xlabel, 
               fontsize = labelsize)
    
    plt.ylabel(ylabel, 
               fontsize = labelsize)
    
    plt.xticks(rotation = rotation)
    
    plt.title(title, 
              fontsize = titlesize,
              loc = 'left')

    plt.tight_layout()

    plt.savefig(os.path.join(graph_output_path,
                f'{file_name}_bp.png'),
               dpi = 300)
    
    plt.show()


#
#import pandas as pd
#
#df = pd.read_csv('sample_dataframe.csv').iloc[:,:3]
#make_boxplot(df,
#             file_name = 'test',
#             xlabel = 'pred',
#             ylabel = 'values',
#             title = 'bp')


#def make_boxplot(data,
#                 file_name,
#                 xlabel,
#                 ylabel,
#                 title):






