import pandas as pd
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn import model_selection 
import os, optuna, joblib
import lightgbm as lgb
import numpy as np


class skfreg_opt():
    
    print('''
          includes stratified KF, 
          hyperparam optimization of
          lightgbm in regression
          ''')

    base_model_path = os.path.join(os.getcwd(),
                                   'output_rs_learn',
                                   'tuned_models')
        
    if not os.path.exists(base_model_path):
        os.makedirs(base_model_path)       
    
    def __init__(
            self, 
            df_train,
            X_cols,
            y_col,
            name,
            num_trials = 10,
            num_folds = 10,
            ):
        
        self.raster_output_path = os.path.join(os.getcwd(), 
                  'output_rs_learn',
                  'tuned_models')

        self.df_train = df_train
        self.X_cols = X_cols
        self.y_col = y_col
        self.num_trials = num_trials
        self.X_data = df_train[X_cols]
        self.y_data = df_train[y_col]
        self.name = name
        self.num_folds = num_folds
               
    def skf_tune(self, trial):
        
        self.df_train["K-FOLD"] = -1   
        self.df_train = self.df_train.sample(frac = 1).reset_index(drop=True)  
        
        num_bins = int(np.floor(1 + np.log2(len(self.df_train))))  
        
        self.df_train.loc[:, "BINS"] = pd.cut(  
            self.df_train[self.y_col], 
            bins = num_bins, 
            labels = False  
        )  

        kf = model_selection.StratifiedKFold(n_splits = self.num_folds,
                                              shuffle = True)          
        
        self.r2_l, self.rmse_l = [], []
        
        for f, (t_, v_) in enumerate(kf.split(X = self.df_train, 
                                      y = self.df_train.BINS.values)):  
            

            self.df_train.loc[v_, 'K-FOLD'] = f
            train = self.df_train[self.df_train['K-FOLD'] != f]
            test = self.df_train[self.df_train['K-FOLD'] == f]
            
            self.X_train, self.X_test, self.y_train, self.y_test = train[self.X_cols], test[self.X_cols], \
                                                train[self.y_col], test[self.y_col] 
            
            
            lgb_train = lgb.Dataset(self.X_train, self.y_train)
            lgb_test = lgb.Dataset(self.X_test, self.y_test)
            
            params = {
                'task': 'train',
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'rf']),
                'metric': 'rmse',
                'objective': trial.suggest_categorical('objective', ['regression_l1', 'regression_l2']),
                'verbosity': 0,
                "seed": 42,
                "learning_rate": trial.suggest_loguniform('learning_rate', 0.05, 1),
                'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
                'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 20),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
                }      
            
            self.model = lgb.train(
                    params, 
                    lgb_train, 
                    valid_names = ['eval', 'train'], 
                    valid_sets = [lgb_test, lgb_train],
                    early_stopping_rounds = 500,
                    num_boost_round  = 10000,
                    )     
            
            pred_test = self.model.predict(self.X_test)
            r2 = r2_score(self.y_test, pred_test)
            rmse = np.sqrt(mean_squared_error(self.y_test, pred_test))
            self.rmse_l.append(rmse) 
            self.r2_l.append(r2)

        return np.mean(self.rmse_l)        
        
    def execute(self,
                X_test_valid, 
                y_test_valid,
                num_folds = 10
                ):

        study = optuna.create_study(direction = 'minimize')
        study.optimize(self.skf_tune, n_trials = self.num_trials)
        
        print('Best trial:')
        trial = study.best_trial
        
        print('  Value: {}'.format(trial.value))
        
        print('  Params: ')
        for key, value in trial.params.items():
            print('    "{}": {},'.format(key, value))
            
            
        model_f = LGBMRegressor(
                boosting_type = trial.params['boosting_type'],
                objective = trial.params['objective'],
                lambda_l1 = trial.params['lambda_l1'],
                lambda_l2 = trial.params['lambda_l2'],
                num_leaves = trial.params['num_leaves'],
                feature_fraction = trial.params['feature_fraction'],
                bagging_fraction = trial.params['bagging_fraction'],
                bagging_freq = trial.params['bagging_freq'],
                min_child_samples = trial.params['min_child_samples'],
                learning_rate = trial.params["learning_rate"]
                ) 
            
        model_f.fit(self.X_data, self.y_data)
        
        joblib.dump(model_f, 
                 os.path.join(
                         os.getcwd(),
                               'output_rs_learn',
                               'tuned_models',
                               f'{self.name}_skfreg_opt_fs.sav'))
        
        scores_fold_r2 = cross_val_score(model_f, 
                      self.X_data, 
                      self.y_data, 
                      cv = num_folds,
                      scoring = 'r2',
                      verbose = 1)
        
        scores_fold_rmse = cross_val_score(model_f, 
              self.X_data, 
              self.y_data, 
              cv = num_folds,
              scoring = 'neg_root_mean_squared_error',
              verbose = 1) 

        print()        
        print(f'R2 per fold of best trial on train data: {scores_fold_r2}')
        print()
        print(f'Mean R2 of folds of best trial on train data: {np.mean(scores_fold_r2)}')

        print()
        print(f'RMSE per fold of best trial on train data: {[x * -1 for x in scores_fold_rmse]}')
        print()
        print(f'Mean RMSE of fold of best trial on train data: {np.mean(scores_fold_rmse)*-1}')

        
        pred_valid = self.model.predict(X_test_valid)                        
        df_predictions = pd.DataFrame(pred_valid).set_index(X_test_valid.index)
        
        r2 = r2_score(y_test_valid, pred_valid)
        rmse = np.sqrt(mean_squared_error(y_test_valid, pred_valid))
        
        print()
        print(f'R2 of test data: {r2}')
        print(f'RMSE of test data: {rmse}')
        print()
        
        return trial.params, df_predictions
        
        
# aa = [-1, -2, -3]
            
# df = pd.read_csv('tanglaw_data.csv')
# # df = pd.read_csv(r"C:\Users\Dlaniger\Projects\MECO_TECO\paper\2020-05-07\final\df_train_final.csv")
# X = df.iloc[:,:-1]
# y = df.iloc[:,-1]
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
# df_train = pd.concat([X_train, y_train], axis = 1)

# optimize = skfreg_opt(
#             df_train = df_train,
#             X_cols = X_train.columns,
#             y_col = 'Chl-a',
#             num_trials = 30,
#             num_folds = 10,
#             name = 'test_opt'
#             )

# model_params, df_predictions = optimize.execute(
#                                                 X_test_valid = X_test, 
#                                                 y_test_valid = y_test,
#                                                 num_folds = 100 #loo method len(y_train)
#                                                 )



# model_f = LGBMRegressor(
#         boosting_type = 'gbdt',
#         objective = 'regression',
#         lambda_l1 = model_params['lambda_l1'],
#         lambda_l2 = model_params['lambda_l2'],
#         num_leaves = model_params['num_leaves'],
#         feature_fraction = model_params['feature_fraction'],
#         bagging_fraction = model_params['bagging_fraction'],
#         bagging_freq = model_params['bagging_freq'],
#         min_child_samples = model_params['min_child_samples']
#         ) 

# model_def = LGBMRegressor() 


# model_def.fit(X_train,y_train)
# print('------------')
# print(f'default train: {model_def.score(X_train, y_train)}')
# print()
# print(f'default test: {model_def.score(X_test, y_test)}')

# model_f.fit(X_train[reduced_X.columns], y_train)
# print()
# print(f'tuned train: {model_f.score(X_test[reduced_X.columns], y_test)}')

# mean_cross_val = np.mean(cross_val_score(model_f, X_train, y_train, cv = 10,
#                                     scoring = 'r2'))
# print(f'tuned test: {mean_cross_val}')

