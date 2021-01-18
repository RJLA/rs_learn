from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def remove_outliers(X,
                    y,
                    max_trials,
                    min_samples,
                    x_cols,
                    xlabel,
                    ylabel,
                    file_name,
                    save_csv = False):

    misc_output_path = os.path.join(os.getcwd(), 
                      'output_rs_learn',
                      'misc')     

    if not os.path.exists(misc_output_path):
        os.makedirs(misc_output_path)

    graph_output_path = os.path.join(os.getcwd(), 
                    'output_rs_learn',
                    'graphs')
        
    if not os.path.exists(graph_output_path):
        os.makedirs(graph_output_path)  

    #normalize data
    X_scaled = MinMaxScaler().fit_transform(X) 

    # remove outliers using ransac estimator
    ransac = RANSACRegressor(base_estimator = LinearRegression(), 
                            max_trials = max_trials,
                            min_samples = min_samples)

    #fit using ransac
    ransac.fit(X_scaled,
            y)

    #select inlier and outlier mask index
    inliers = ransac.inlier_mask_
    outliers = np.logical_not(inliers)

    #select inlier and outlier
    X_inlier = X[inliers]
    y_inlier = y[inliers]
    X_outlier = X[outliers]
    y_outlier = y[outliers]

    #merge inlier
    df_inlier = pd.concat([X_inlier,
                        y_inlier],
                        axis = 1)

    df_inlier.reset_index(inplace = True)

    del df_inlier['index']

    # merge outlier data
    df_outlier = pd.concat([X_outlier,
                        y_outlier],
                        axis = 1)

    df_outlier.reset_index(inplace = True)

    del df_outlier['index']

    if save_csv is True:
        df_inlier.to_csv(os.path.join(misc_output_path,
        'inliers.csv'), index = False)

    #show inlier and outliers

    f, ax = plt.subplots(figsize = (8,8))

    plt.scatter(X_outlier[x_cols],
                y_outlier,
                linewidth = 1,
                facecolors = 'None',
                marker = 's',
                label = 'Outliers',
                color = 'k')

    plt.scatter(X_inlier[x_cols],
                y_inlier,
                linewidth = 1,
                facecolors = 'None',
                color = 'r',
                marker = 's',
                label = 'Inliers')

    plt.xlabel(xlabel,
            fontsize = 10)

    plt.ylabel(ylabel,
            fontsize = 10)

    plt.title('Inliers vs Outliers',
                fontsize = 12,
                loc = 'left')

    plt.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(graph_output_path,
                'Scatter_inlier_vs_outlier_%s.png'%file_name),
                dpi = 300)
    
    return df_inlier, df_outlier
