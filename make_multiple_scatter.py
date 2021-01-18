import os
import matplotlib.pyplot as plt

def make_multiple_scatter(X,
                          y,
                          file_name,
                          title,
                          color_list):

    graph_output_path = os.path.join(os.getcwd(), 
                      'output_rs_learn',
                      'graphs')

    if not os.path.exists(graph_output_path):
        os.makedirs(graph_output_path)
    
    print('''
    args: X, y, file_name,
    title, color_list,
          ''')
    print('''
    more color codes in this site
    https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    ''')

    f, ax = plt.subplots(figsize = (8,8))
        
    for i,j in zip(X.columns,
                   color_list):
        
        ax.scatter(X[i],
                   y,
                   label = '%s'%i,
                   color = j,
                   marker = 's',
                  facecolor = 'None')
    
    plt.xlabel(i,
             fontsize = 10)
    
    plt.ylabel(y.name,
             fontsize = 10)

    plt.title(title,
                loc = 'left',
                fontsize = 12)
    
    plt.legend(loc = 'best',
              framealpha = 1)
              
    plt.savefig(os.path.join(graph_output_path, 
    'Multiple_scatter_%s.png'%file_name),
               dpi = 300)
    plt.show()