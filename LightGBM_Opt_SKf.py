import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from sklearn import model_selection 
import os
import optuna
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import numpy as np

class LightGBM_Opt_SKf():
             
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
        
        model_l = []
        
        kf = model_selection.StratifiedKFold(n_splits = self.fold)          
        
        rmse_l = []
        
        for f, (t_, v_) in enumerate(kf.split(X = self.df, 
                                      y = self.df.BINS.values)):  

            self.df.loc[v_, 'K-FOLD'] = f
            train = self.df[self.df['K-FOLD'] != f]
            test = self.df[self.df['K-FOLD'] == f]
            
            X_train, X_test, y_train, y_test = train[self.X_cols], test[self.X_cols], \
                                                train[self.y_col], test[self.y_col] 
    
            X_train = pd.DataFrame(StandardScaler().fit_transform(X_train), 
                              columns = X_train.columns).set_index(X_train.index)
            
            X_test = pd.DataFrame(StandardScaler().fit_transform(X_test), 
                              columns = X_test.columns).set_index(X_test.index)    
                
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_test = lgb.Dataset(X_test, y_test)

        
            params = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': {'l2'},
                'verbosity': -1,
                "seed":42,
                "learning_rate":trial.suggest_loguniform('learning_rate', 0.005, 0.03),
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
                    num_boost_round = 500,
                    valid_names=["train", "valid"], 
                    valid_sets = [lgb_train, lgb_test],
                    early_stopping_rounds = 10,
                    verbose_eval = 0
                    )

            actual = y_test
            predicted = model.predict(X_test)
                    
            rmse_ = sqrt(mean_squared_error(actual, predicted))
            
            rmse_l.append(rmse_)
            model_l.append(model)
            
            print(f'Fold {f}: {rmse_}')
        
        print(f'Ave. rmse {np.mean(rmse_l)}')
        print()
        return np.mean(rmse_l)    
        
        
    def execute(self):
        study = optuna.create_study(direction = 'minimize')

        study.optimize(self.skf_tune, n_trials = 30)
        
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
           
        pipeline = Pipeline(
                [
                    ('sc', StandardScaler()),
                    ('mod', model_f)
                ]
                )
        
        pipeline.fit(self.X_data,
                    self.y_data)        

        joblib.dump(pipeline, 
                    os.path.join(
                            os.getcwd(),
                                 'output_rs_learn',
                                 'tuned_models',
                                 f'{self.name}_optimized_opt.sav')
                    )
        return trial.params
            
            
#df = pd.read_csv('train.csv')
#X = df.iloc[:,:-1]
#y = df.iloc[:,-1]
#optimize = LightGBM_Opt_SKf(
#            df,
#            X_cols = X.columns,
#            y_col = 'LST',
#            fold = 2,
#            name = 'test_opt'
#            )
#model_params = optimize.execute()
#
#df_test = pd.read_csv('test.csv')
#X = df_test.iloc[:,:-1]
#y = df_test.iloc[:,-1]
#test_mod = joblib.load(r'C:\Users\Dlaniger\Anaconda3\Lib\site-packages\rs_learn\output_rs_learn\tuned_models\test_opt_optimized_opt.sav')
#
#aa = test_mod.predict(X)
#rmse = sqrt(mean_squared_error(y, aa))
#print(rmse)