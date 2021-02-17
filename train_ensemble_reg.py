import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
import joblib

def train_ensemble_reg(X_train,
                   y_train,
                   key_list,
                   model_list,
                   param_list,
                   name,
                   ws,
                   folds = 10,
                   n_iteration = 1,):
    
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
    
    def cross_validate_tune(key,
                       model,
                       n,
                       params):
    
        pipeline = Pipeline([('sc', 
                      MinMaxScaler()),
                    (f'{key}',
                     model)])
    
        kf = KFold(n_splits = folds, 
                   shuffle = True) 
        
        rscv = RandomizedSearchCV(estimator = pipeline,
                      param_distributions = params,
                      scoring = 'r2', 
                      n_jobs = -1, 
                      cv = kf,
                      verbose = True)
        
        # scores_fold = [absolute(e) for e in scores_fold]
       
        
        # mean_kf = sum(scores_fold) / len(scores_fold) 
    
        # pipeline.fit(X_train, 
        #              y_train)


        rscv.fit(X_train, y_train)
        print(f'Best score: {rscv.best_score_}')
        print(f'Best params: {rscv.best_params_}')
        print(f'Best params: {rscv.best_estimator_}')
        
        model_output = f'{name}_{n}_{key}.sav'
        
        print(f'Model save: {model_output}')
            
        joblib.dump(rscv, 
                    os.path.join(base_model_path,
                                 model_output))
        
        # with open(os.path.join(misc_output_path,
        #                        f'kfold_{name}_{n}_{key}.txt'),"w") as t:
            
        #     result = f'Scores per fold: {scores_fold}\nAverage: %.4f'%mean_kf
        #     t.write(result)
        #     t.close()
            
        # print(f'Scores mean kfold: {mean_kf}')
        # print(f'Scores per fold: {scores_fold}')
        
    # prediction_list = []
    # model_name_list = []
    
    for n in range(n_iteration):

        for key, model, params in zip(key_list,
                              model_list,
                              param_list):
            print()
            print(f'Number of iterations {n}')
            print(f'Model {key}:' )
            
            cross_validate_tune(key,
                          model, n, params)
    #         model_name = f'{name}_{n}_{key}'
    #         model_trained = joblib.load(os.path.join(base_model_path, 
    #                                                  f'{model_name}.sav'))

    #         prediction = model_trained.predict(X_train)
    #         prediction_list.append(prediction)
    #         model_name_list.append(model_name)

    # df_predictions = pd.DataFrame(prediction_list).T
    # df_predictions.columns = model_name_list
    
    # return df_predictions
    
#list of regressors to be used


# from lightgbm import LGBMRegressor
# from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
# import numpy as np
# df = pd.read_csv('sample_data.csv')
# X_train =  df.iloc[:,:-1]
# y_train = df.iloc[:,-1]

# # oss_range = ['ls', 'lad', 'huber', 'quantile']
# # learning_rate_range = [0.001, 0.01, 0.1]
# # n_estimators_range = [100, 500, 1000]
# # subsample_range = np.arange(0.5,2.1,0.1)
# # min_samples_split_range = np.arange(2,11)
# # min_samples_leaf_range = np.arange(1,11)
# # max_depth_range = np.arange(3,11)

# # param_grid_dct = [{
# #                 'gb__loss': loss_range, 
# #                 'gb__learning_rate': learning_rate_range,
# #                 'gb__n_estimators': n_estimators_range,
# #                 'gb__subsample': subsample_range,
# #                 'gb__min_samples_split': min_samples_split_range,
# #                 'gb__min_samples_leaf': min_samples_leaf_range,
# #                 'gb__max_depth': max_depth_range
# #                  }]

# key_list = [
#     'lgb',
#     'xgb',
#     'cat'
#     ]

# model_list = [
#     LGBMRegressor(n_estimators = 200, n_jobs = -1,),
#     XGBRegressor(n_estimators = 1000,
#                             n_jobs = -1,
#                             objective='reg:squarederror'),
#     CatBoostRegressor(silent = True)
#     ]


# param_grid_dct_list = [
#     {
#         'lgb__boosting_type':['gbdt', 'rf', 'dart', 'goss'],
#         'lgb__learning_rate':np.arange(0,1.05,0.01),
#         'lgb__reg_alpha':np.arange(0,1.05,0.01),
#         'lgb__reg_lambda':np.arange(0,1.05,0.01),
#          'lgb__num_leaves':np.arange(1,100,1, dtype = int),
#      },
    
#     {
#         'xgb__boosting_type':['gbdt', 'rf', 'dart', 'goss'],
#         'xgb__learning_rate':np.arange(0,1.05,0.01),
#         'xgb__reg_alpha':np.arange(0,1.05,0.01),
#         'xgb__reg_lambda':np.arange(0,1.05,0.01),
#         'xgb__num_leaves':np.arange(1,100,1, dtype = int),
#      },
    
#     # {
#     #   'cat__depth':np.arange(1,11,1),
#     #   'cat__iterations':[250,100,500,1000],
#     #   'cat__learning_rate':np.arange(0,1.05,0.05), 
#     #   'cat__l2_leaf_reg':[3,1,5,10,100],
#     #   'cat__border_count':[32,5,10,20,50,100,200],
#     #   'cat__ctr_border_count':[50,5,10,20,100,200],
#     #   'cat__thread_count':4} 
#     ]

# prediction_train_l1 = train_ensemble_reg(X_train = X_train,
#                     y_train = y_train,
#                     key_list = key_list,
#                     model_list = model_list,
#                     param_grid_dct_list = param_grid_dct_list,
#                     name = 'level_1_test',
#                     ws = os.getcwd(),
#                     folds = 10,
#                     n_iteration = 1)