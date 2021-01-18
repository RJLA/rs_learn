import pandas as pd
from statistics import mean
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import os

def select_features_gd(
        X_train,
        y_train,
        file_name,
        model = GradientBoostingRegressor(n_estimators = 100,
                                          verbose = 0),
        scaler = StandardScaler(),
        figsize = (20, 20),
        fontsize_ticks = 20,
        labelsize = 15,
        legendsize = 15,
        titlesize = 15,
        scoring = 'r2',
        threshold = 'mean'
        ):

    
    graph_output_path = os.path.join(os.getcwd(), 
                      'output_rs_learn',
                      'graphs')

    if not os.path.exists(graph_output_path):
        os.makedirs(graph_output_path)
        
    std_train = pd.DataFrame(StandardScaler().fit_transform(X_train), 
                               columns = X_train.columns).set_index(X_train.index)
    model.fit(std_train, 
                   y_train)
        
    model_imp_feat = SelectFromModel(model, 
                            prefit = True,
                            threshold = threshold)

    #make list of important features
    imp_feat = [x for x, 
                  y in zip(X_train.columns, 
                           model_imp_feat.get_support()) if y == True]

    reduced_X = pd.DataFrame(model_imp_feat.transform(std_train),
                             columns = imp_feat).set_index(std_train.index)
    
    model_imp = model.fit(reduced_X, 
                          y_train)
    
#    #Plot feature importance
    feature_importance = permutation_importance(model_imp, 
                                                reduced_X, 
                                                y_train, 
                                                n_repeats = 100,
                                random_state = 42)    
    

    perm_sorted_idx = feature_importance.importances_mean.argsort()
    

    tree_importance_sorted_idx = np.argsort(model_imp.feature_importances_)

    
    tree_indices = np.arange(0, len(model_imp.feature_importances_)) + 0.5
    
    fig, (ax1, ax2) = plt.subplots(1, 2, 
         figsize = figsize)
    
    ax1.barh(tree_indices,
             model_imp.feature_importances_[tree_importance_sorted_idx], 
             height = 0.7,
             color = 'grey')

    ax1.set_yticklabels(reduced_X.columns[tree_importance_sorted_idx],
                            fontsize = fontsize_ticks)
    ax1.set_yticks(tree_indices)
    
    ax1.set_ylim((0, len(model_imp.feature_importances_)))
    
    labels = reduced_X.columns[perm_sorted_idx]
    
    ax2.boxplot(feature_importance.importances[perm_sorted_idx].T, 
                vert = False,
            labels = labels)
    
    ax2.set_yticklabels(reduced_X.columns[perm_sorted_idx],
                        fontsize = fontsize_ticks)
    
    df_imp = pd.DataFrame(feature_importance.importances[perm_sorted_idx])
    df_imp['sum_imp'] = df_imp.sum(axis = 1)
    df_imp['features'] = reduced_X.columns[perm_sorted_idx]
    df_imp['mean'] = df_imp.iloc[:,:-2].mean(axis = 1)
    feat_mean = df_imp['mean'].mean(axis = 0)

    ax2.axvline(feat_mean , 
        color = 'k', 
        linestyle = '--', 
        linewidth = 1,
        label = 'PD Mean')
    
    ax2.axvline(0, 
        color = 'r', 
        linestyle = '--', 
        linewidth = 1,
        label = 'Zero line')
    
    ax2.legend(fontsize = legendsize)
    

    ax1.set_xlabel('Impurity',
               fontsize = labelsize)
    
    ax2.set_xlabel('Performance deterioration',
           fontsize = labelsize)
    
    ax1.set_ylabel('Features',
               fontsize = labelsize)
    
    ax1.set_title('Gini importance ranking',
              loc = 'left',
              fontsize = titlesize) 
    
    ax2.set_title('Permutation importance ranking',
              loc = 'left',
              fontsize = titlesize)     
    fig.tight_layout()
    
    fig.savefig(os.path.join(graph_output_path,
                f'{file_name}_var_imp.png'),
               dpi = 300)    

    plt.show()  
    
    pipeline = Pipeline([('sc', 
                  scaler),
                ('mod',
                 model)])
    
    kf = KFold(n_splits = 10, 
               shuffle = True) 

    scores_fold = cross_val_score(pipeline, 
                                  X_train[labels], 
                                  y_train, 
                                  cv = kf,
                                  scoring = scoring,
                                  verbose = 1)

    if scoring is 'neg_root_mean_squared_error':        
        print(f'Score using important features: {[-1 * i for i in scores_fold]}')
        print(f'Average scores of 10 folds {-1 * mean(scores_fold)}')
        
    else:
        print(f'Score using important features: {scores_fold}')
        print(f'Average scores of 10 folds {mean(scores_fold)}')
    
    print(f'Number of original features: {X_train.shape[1]}')
    print(f'Number of selected features: {len(labels)}')
    print(f'Important features selected: {list(labels)}')

    return labels, df_imp 
#
#df = pd.read_csv('sample_dataframe.csv')
#X =  df.iloc[:,:-2]
#y = df.iloc[:,-2]
###
#aa,bb = select_features_gd(X,y,'test',figsize = (10,10),
#    fontsize_ticks = 25,
#    labelsize = 20,
#    titlesize = 18,    
#    threshold = 'mean')

#select_features_gd(X,y,'test',figsize = (10,10),
#    fontsize_ticks = 25,
#    labelsize = 20,
#    titlesize = 18,   
#    scoring = 'neg_root_mean_squared_error' ,
#    threshold = 'mean')

#select_features_gd(X,y,'test',figsize = (10,10),
#    fontsize_ticks = 25,
#    labelsize = 20,
#    titlesize = 18,   
#    scoring = 'r2' ,
#    threshold = 'mean')

