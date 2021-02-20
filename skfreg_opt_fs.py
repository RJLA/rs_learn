import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMRegressor
from sklearn import model_selection 
from sklearn.metrics import r2_score
import os
import optuna
import joblib
import lightgbm as lgb
import numpy as np

class skfreg_opt_fs():
    print('''
          includes stratified KF, 
          hyperparam optimization, 
          lightgbm in regression
          ''')

             
    base_model_path = os.path.join(os.getcwd(),
                                   'output_rs_learn',
                                   'tuned_models')
        
    if not os.path.exists(base_model_path):
        os.makedirs(base_model_path)       

    
    def __init__(
            self, 
            df,
            X_cols,
            y_col,
            name,
            fold = 10,
            ):
        
        self.raster_output_path = os.path.join(os.getcwd(), 
                  'output_rs_learn',
                  'tuned_models')

        self.df = df
        self.X_cols = X_cols
        self.y_col = y_col
        self.fold = fold
        self.X_data = df[X_cols]
        self.y_data = df[y_col]
        self.name = name
        
    def skf_tune(self, trial):
        
        self.df["K-FOLD"] = -1   
        self.df = self.df.sample(frac = 1).reset_index(drop=True)  
        
        num_bins = int(np.floor(1 + np.log2(len(self.df))))  
        
        self.df.loc[:, "BINS"] = pd.cut(  
            self.df[self.y_col], 
            bins = num_bins, 
            labels = False  
        )  
        

        kf = model_selection.StratifiedKFold(n_splits = self.fold)          
        
        rmse_l, r2_l = [],[]
        model_l = []
        
        for f, (t_, v_) in enumerate(kf.split(X = self.df, 
                                      y = self.df.BINS.values)):  

            self.df.loc[v_, 'K-FOLD'] = f
            train = self.df[self.df['K-FOLD'] != f]
            test = self.df[self.df['K-FOLD'] == f]
            
            self.X_train, self.X_test, self.y_train, self.y_test = train[self.X_cols], test[self.X_cols], \
                                                train[self.y_col], test[self.y_col] 
    
            
            lgb_train = lgb.Dataset(self.X_train, self.y_train)
            lgb_test = lgb.Dataset(self.X_test, self.y_test)
            
            params = {
                'task': 'train',

                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'rf']),
                'objective': 'regression',
                # 'metric': {'l2'},
                'metric': trial.suggest_categorical('metric', ['l1', 'l2']),
                'verbosity': -1,
                "seed": 42,
                "learning_rate": trial.suggest_loguniform('learning_rate', 0.05, 1),
                'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            }        
            
            model = lgb.train(
                    params, 
                    lgb_train, 
                    num_boost_round = 1000,
                    valid_names=["train", "valid"], 
                    valid_sets = [lgb_train, lgb_test],
                    early_stopping_rounds = 100,
                    verbose_eval = 0
                    )
    

            actual = self.y_test
            predicted = model.predict(self.X_test)
                    
            rmse_ = sqrt(mean_squared_error(actual, predicted))
            r2_ = r2_score(actual, predicted)
            
            rmse_l.append(rmse_)
            r2_l.append(r2_)
            model_l.append(model)
            
        return np.mean(rmse_l)            
        
    def execute(self):
        
        study = optuna.create_study(direction = 'minimize')

        study.optimize(self.skf_tune, n_trials = 5)
        
        print('Best trial:')
        trial = study.best_trial
        
        print('  Value: {}'.format(trial.value))
        
        print('  Params: ')
        for key, value in trial.params.items():
            print('    "{}": {},'.format(key, value))

        model_f = LGBMRegressor(
                boosting_type = 'gbdt',
                objective = 'regression',
                lambda_l1 = trial.params['lambda_l1'],
                lambda_l2 = trial.params['lambda_l2'],
                num_leaves = trial.params['num_leaves'],
                feature_fraction = trial.params['feature_fraction'],
                bagging_fraction = trial.params['bagging_fraction'],
                bagging_freq = trial.params['bagging_freq'],
                min_child_samples = trial.params['min_child_samples']
                ) 
        
        model_f.fit(self.X_data, self.y_data)  
        
        model_imp_feat = SelectFromModel(model_f, 
                                          prefit = True,
                                          threshold = None)
            
        imp_feat = [x for x, 
                      y in zip(self.X_data.columns, 
                               model_imp_feat.get_support()) if y == True]
    
        reduced_X = pd.DataFrame(model_imp_feat.transform(self.X_data),
                                 columns = imp_feat).set_index(self.X_data.index)
        
        model_f.fit(reduced_X, self.y_data)   
        
        predictions = model_f.predict(reduced_X)
        
        df_predictions = pd.DataFrame(predictions).set_index(self.X_data.index)
        

        joblib.dump(model_f, 
                    os.path.join(
                            os.getcwd(),
                                 'output_rs_learn',
                                 'tuned_models',
                                 f'{self.name}_skfreg_opt_fs.sav'))
        
        print(f'Number of original features: {self.X_train.shape[1]}')
        print(f'Number of selected features: {len(imp_feat)}')
        print(f'Important features selected: {list(imp_feat)}')
    
        return trial.params, reduced_X, df_predictions
            
            
# # df = pd.read_csv('tanglaw_data.csv')
# df = pd.read_csv(r"C:\Users\Dlaniger\Projects\MECO_TECO\paper\2020-05-07\final\df_train_final.csv")
# X = df.iloc[:,:-1]
# y = df.iloc[:,-1]
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
# df_train = pd.concat([X_train, y_train],axis = 1)

# optimize = skfreg_opt_fs(
#             df_train,
#             X_cols = X_train.columns,
#             y_col = 'Chl-a',
#             fold = 10,
#             name = 'test_opt'
#             )
# model_params, reduced_X, df_predictions = optimize.execute()

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

