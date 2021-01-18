import os, joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

def cross_validate_classification(X_train,
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
                  MinMaxScaler()),
                (f'{key}',
                 model)])
            
    def make_cross(X_train_skf,
                   y_train_skf,
                   model_type):
        
        scores_fold = cross_val_score(model_type, 
                              X_train_skf, 
                              y_train_skf, 
                              cv = skf,
                              scoring = scoring,
                              verbose = 1)
        
        mean_kf = sum(scores_fold) / len(scores_fold) 
    
        model_type.fit(X_train_skf, 
                     y_train_skf)
        
        joblib.dump(model_type, 
                    os.path.join(base_model_path,
                                 f'{name}_{key}.sav'))
        
        with open(os.path.join(misc_output_path,
                               f'kfold_{name}_{key}.txt'),"w") as t:
            
            result = f'Scores per fold: {scores_fold}\nAverage: %.4f'%mean_kf
            t.write(result)
            t.close()
            
        print(f'Scores mean kfold: {mean_kf}')
        print(f'Scores per fold: {scores_fold}')        
        
        return model_type    
  
    skf = StratifiedKFold(n_splits = folds,
                          shuffle = True)
    
    for train_index, test_index in skf.split(X_train, 
                                             y_train):
        
        X_train_skf, y_train_skf = X_train.iloc[train_index], y_train[train_index]
        
        if tune is True:
            print('Tuning model..')
            print()
            if len(params) == 0:
                print('Parameter dict must not be zero. Program exited')
                print('Input parameter dict or use tune = False instead')
                return None
            else:
                gs = RandomizedSearchCV(estimator = pipeline,
                                  param_distributions = params,
                                  scoring = scoring, 
                                  n_jobs = -1, 
                                  cv = 10,
                                  verbose = True)
                
                gs.fit(X_train_skf, 
                       y_train_skf)
                
                best = gs.best_estimator_
                
            model_final = make_cross(X_train_skf,
                                     y_train_skf,
                                     model_type = best)
            
        elif tune is False:
            print('tune = False')
            print('Model will not be tuned. Using default model settings')
            print()
            model_final = make_cross(X_train_skf,
                                     y_train_skf,
                                     pipeline)
        
    
        return model_final    
    
    
#import pandas as pd
#from lightgbm import LGBMClassifier
#from sklearn.preprocessing import LabelEncoder
#
#df = pd.read_csv(r"C:\Users\Dlaniger\Anaconda3\Lib\site-packages\rs_learn\sample_dataframe.csv")
#X, y = df.iloc[:,:-2], df.iloc[:,-1]
#lec = LabelEncoder()
#y_enc = lec.fit_transform(y)
#
## tune model
#params_lgbm = {'lgbm__max_depth': range(1,10,1),
#     'lgbm__num_leaves': [int((2 ** i)/2) for i in range(2,10,1)]}
#
#model_lgbm = LGBMClassifier(n_estimators = 1000,
#                      min_data_leaf = 1000,
#                      boosting_type = 'dart')
##untuned
##cross_validate_classification(X_train = X,
##                   y_train = y_enc,
##                   model = model_lgbm,
##                   key = 'lgbm',
##                   scoring = 'f1_weighted',
##                   name = 'test',
##                   folds = 10,
##                   tune = False,
##                   params = {})   
#    
#
#cross_validate_classification(X_train = X,
#                   y_train = y_enc,
#                   model = model_lgbm,
#                   key = 'lgbm',
#                   scoring = 'f1_weighted',
#                   name = 'test',
#                   folds = 10,
#                   tune = True,
#                   params = params_lgbm) 
#    
    
    
    
    
    