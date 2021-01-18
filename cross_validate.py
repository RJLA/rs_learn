import os, sys, joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def cross_validate(X_train,
                   y_train,
                   model,
                   key,
                   scoring,
                   name,
                   folds = 10,
                   tune = False,
                   params = {}):
        
    base_model_path = os.path.join(os.getcwd(),
                                   'output_rs_learn',
                                   'tuned_models')

    if not os.path.exists(base_model_path):
        os.makedirs(base_model_path)
        
    misc_output_path = os.path.join(os.getcwd(), 
                      'output_rs_learn',
                      'misc')     

    if not os.path.exists(misc_output_path):
        os.makedirs(misc_output_path)    
        
    pipeline = Pipeline([('sc', 
                  StandardScaler()),
                ('%s'%key,
                 model)])

    kf = KFold(n_splits = folds, 
               shuffle = True) 
    
    def make_cross(model_type):
        scores_fold = cross_val_score(model_type, 
                              X_train, 
                              y_train, 
                              cv = kf,
                             scoring = scoring,
                             verbose = 1)
        
        mean_kf = sum(scores_fold) / len(scores_fold) 
    
        model_type.fit(X_train, 
                     y_train)
        
            
        joblib.dump(model_type, 
                    os.path.join(base_model_path,
                                 '%s_%s.sav'%(name, key)))
        
        with open(os.path.join(misc_output_path,
                               'kfold_%s_%s.txt'%(name, key)),"w") as t:
            
            result = 'Scores per fold: %s\nAverage: %.4f'%(scores_fold, mean_kf)
            t.write(result)
            t.close()
            
        print('Scores mean kfold: %s' %mean_kf)
        print('Scores per fold: %s' %scores_fold)  
        print(model_type)
        
        return model_type
    
    if tune is True:
        print('Tuning model..')
        gs = RandomizedSearchCV(estimator = pipeline,
                              param_distributions = params,
                              scoring = scoring, 
                              n_jobs = -1, 
                              cv = 10,
                              verbose = True)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_            
        model_final = make_cross(model_type = best)
        
    elif tune is False:
        print('tune = False')
        print('Model will not be tuned. Using default model settings')
        print()
        model_final = make_cross(pipeline)
    

    return model_final

#import pandas as pd
#from lightgbm import LGBMRegressor
#
#
#df = pd.read_csv('sample_dataframe.csv')
#X = df.iloc[:,:-2]
#y = df.iloc[:,-2]
#
## # tune model
#params_lgbm = {'lgbm__max_depth': range(1,10,1),
#     'lgbm__num_leaves': [int((2 ** i)/2) for i in range(2,10,1)]}
#
#model_lgbm = LGBMRegressor(n_estimators = 1000,
#                      min_data_leaf = 1000,
#                      boosting_type = 'dart',
#                     learning_rate = 1E-1)
## X_train[important_feat]
#model_cv = cross_validate(X,
#                   y,
#                   model = model_lgbm,
#                   key = 'lgbm',
#                   scoring = 'neg_root_mean_squared_error',
#                   name = '2020-14-06',
#                   folds = 10,
#                             tune = False,
#                             params = {})