import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn import model_selection 
from xgboost import XGBRegressor
import os, optuna, joblib
import numpy as np


class skf_xgbr_opt():
    
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
            scaler = StandardScaler()
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
        
        self.sc = scaler
               
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
                                                
            self.X_train_sc, self.X_test_sc = self.sc.fit_transform(self.X_train), self.sc.transform(self.X_test)
            
            dmatrix_train = xgb.DMatrix(data = self.X_train_sc, label = self.y_train)
            dmatrix_test = xgb.DMatrix(data = self.X_test_sc, label = self.y_test)
            
            params = {
                    "silent": 1,
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                    "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
                    "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
                    }

            if params["booster"] == "gbtree" or params["booster"] == "dart":
                params["max_depth"] = trial.suggest_int("max_depth", 1, 9)
                params["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
                params["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
                params["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
            if params["booster"] == "dart":
                params["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
                params["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
                params["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
                params["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)
                
            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-rmse")
            
            self.model = xgb.train(params, 
                            dmatrix_train, 
                            evals = [(dmatrix_test, "validation")], 
                            callbacks = [pruning_callback]
                            )
            
            dmatrix_test = xgb.DMatrix(data = self.X_test_sc, label = self.y_test)
            pred_test = self.model.predict(dmatrix_test)
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
            
            
        model_f = XGBRegressor(**trial.params) 
        
        pipeline = Pipeline([('sc', self.sc),
                              ('model', model_f)])
        
        pipeline.fit(self.X_data, self.y_data)
        
        joblib.dump(pipeline, 
                  os.path.join(
                          os.getcwd(),
                                'output_rs_learn',
                                'tuned_models',
                                f'{self.name}_skf_xgbr_opt.sav'))
        
        scores_fold_r2 = cross_val_score(pipeline, 
                      self.X_data , 
                      self.y_data, 
                      cv = num_folds,
                      scoring = 'r2',
                      verbose = 1)
        
        scores_fold_rmse = cross_val_score(pipeline, 
              self.X_data , 
              self.y_data, 
              cv = num_folds,
              scoring = 'neg_root_mean_squared_error',
              verbose = 1) 
        
        r2_train_res_1  = f'R2 per fold of best trial on train data: {scores_fold_r2}'
        r2_train_res_2 = f'Mean R2 of folds of best trial on train data: {np.mean(scores_fold_r2)}'
        rmse_train_1 = f'RMSE per fold of best trial on train data: {[x * -1 for x in scores_fold_rmse]}'
        rmse_train_2 = f'Mean RMSE of fold of best trial on train data: {np.mean(scores_fold_rmse)*-1}'

        print()        
        print(r2_train_res_1)
        print()
        print(r2_train_res_2)

        print()
        print(rmse_train_1)
        print()
        print(rmse_train_2)

        pred_valid = pipeline.predict(X_test_valid)                        
        df_predictions = pd.DataFrame(pred_valid).set_index(X_test_valid.index)
        
        r2 = r2_score(y_test_valid, pred_valid)
        rmse = np.sqrt(mean_squared_error(y_test_valid, pred_valid))
        
        r2_test = f'R2 of test data: {r2}'
        rmse_test = f'RMSE of test data: {rmse}'
        print()
        print(r2_test)
        print()
        print(rmse_test)
        
        return trial.params, df_predictions, r2_train_res_1, r2_train_res_2, rmse_train_1, rmse_train_2, r2_test, rmse_test
        
        

            
# df = pd.read_csv('tanglaw_data.csv')
# # df = pd.read_csv(r"C:\Users\Dlaniger\Projects\MECO_TECO\paper\2020-05-07\final\df_train_final.csv")
# X = df.iloc[:,:-1]
# y = df.iloc[:,-1]
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
# df_train = pd.concat([X_train, y_train], axis = 1)

# optimize = skf_xgbr_opt(
#             df_train = df_train,
#             X_cols = X_train.columns,
#             y_col = 'Chl-a',
#             num_trials = 2,
#             num_folds = 2,
#             name = 'test_opt'
#             )

# model_params, df_predictions, r2_train_res_1, r2_train_res_2, rmse_train_1, rmse_train_2, r2_test, rmse_test = optimize.execute(
#                                                 X_test_valid = X_test, 
#                                                 y_test_valid = y_test,
#                                                 num_folds = 2 #loo method len(y_train)
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

