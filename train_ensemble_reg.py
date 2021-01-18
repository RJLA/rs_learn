import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import pandas as pd
import joblib
from numpy import absolute
from sklearn.model_selection import cross_val_score

def train_ensemble_reg(X_train,
                   y_train,
                   key_list,
                   model_list,
                   name,
                   ws,
                   folds = 10,
                   n_iteration = 1):
    
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
    
    def cross_validate(key,
                       model,
                       n):
    
        pipeline = Pipeline([('sc', 
                      MinMaxScaler()),
                    (f'{key}',
                     model)])
    
        kf = KFold(n_splits = folds, 
                   shuffle = True) 
        
        scores_fold = cross_val_score(pipeline, 
                                      X_train, 
                                      y_train, 
                                      cv = kf,
                                     scoring = 'neg_root_mean_squared_error')
        
        scores_fold = [absolute(e) for e in scores_fold]
       
        
        mean_kf = sum(scores_fold) / len(scores_fold) 
    
        pipeline.fit(X_train, 
                     y_train)

            
        joblib.dump(pipeline, 
                    os.path.join(base_model_path,
                                 f'{name}_{n}_{key}.sav'))
        
        with open(os.path.join(misc_output_path,
                               f'kfold_{name}_{n}_{key}.txt'),"w") as t:
            
            result = f'Scores per fold: {scores_fold}\nAverage: %.4f'%mean_kf
            t.write(result)
            t.close()
            
        print(f'Scores mean kfold: {mean_kf}')
        print(f'Scores per fold: {scores_fold}')
        
    prediction_list = []
    model_name_list = []
    
    for n in range(n_iteration):

        for key, model in zip(key_list,
                             model_list):
            print()
            print(f'Number of iterations {n}')
            print(f'Model {key}:' )
            
            cross_validate(key,
                          model,n)
            model_name = f'{name}_{n}_{key}'
            model_trained = joblib.load(os.path.join(base_model_path, 
                                                     f'{model_name}.sav'))

            prediction = model_trained.predict(X_train)
            prediction_list.append(prediction)
            model_name_list.append(model_name)

    df_predictions = pd.DataFrame(prediction_list).T
    df_predictions.columns = model_name_list
    
    return df_predictions
    