from sklearn import model_selection 
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from math import sqrt

def create_folds(
        data,
                 X_cols,
                 y_col,
                model,
                scaler = StandardScaler(),
                folds = 10
                ):  
    
    data["K-FOLD"] = -1   
    data = data.sample(frac = 1).reset_index(drop=True)  
    num_bins = int(np.floor(1 + np.log2(len(data))))  
    data.loc[:, "BINS"] = pd.cut(  
        data[y_col], 
        bins = num_bins, 
        labels = False  
    )  
    
    kf = model_selection.StratifiedKFold(n_splits = folds)  
    
    pipeline = Pipeline([('sc', 
              scaler),
            ('mod',
             model)])
    
    rmse_fold = []
    nrmse_fold = []
    mnb_fold = []
    nmae_fold = []
    score_fold = []
    
    #fill the new kfold column   
    for f, (t_, v_) in enumerate(kf.split(X = data, 
                                          y = data.bins.values)):  
        data.loc[v_, 'K-FOLD'] = f
        train = data[data['K-FOLD'] != f]
        test = data[data['K-FOLD'] == f]
        X_train, X_test, y_train, y_test = train[X_cols], test[X_cols], \
                                            train[y_col], test[y_col] 
                                            
        pipeline.fit(X_train, y_train)

        per_err =  (np.abs((pipeline.predict(X_test) - y_test)) / y_test) * 100
        
        rmse = sqrt(mean_squared_error(y_test, pipeline.predict(X_test)))
        nrmse = np.std(per_err)
        mnb = np.mean(per_err)                
        nmae = np.mean(np.abs(per_err))
        r2 = pipeline.score(X_test, y_test)
        
        rmse_fold.append(rmse)
        nrmse_fold.append(nrmse)
        mnb_fold.append(mnb)
        nmae_fold.append(nmae)
        score_fold.append(r2)
        
        

    print(f'Results of fold {f+1}:')
    print()
    print(f'RMSE: {rmse_fold}. Ave. fold: {np.mean(rmse_fold)}')
    print()    
    print(f'NRMSE: {nrmse_fold}. Ave. fold: {np.mean(nrmse_fold)}')  
    print()
    print(f'MNB: {mnb_fold}. Ave. fold: {np.mean(mnb_fold)}')  
    print()
    print(f'NMAE: {nmae_fold}. Ave. fold: {np.mean(nmae_fold)}')  
    print()
    print(f'R2: {score_fold}. Ave. fold: {np.mean(score_fold)}')  
    


from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv('sample_dataframe.csv')
X = df.iloc[:,:-2]
y = df.iloc[:,-2]

model = GradientBoostingRegressor(n_estimators = 500, 
                                  verbose = 0)

create_folds(df,
             X_cols = list(X.columns),
             y_col = 'y_real',
             model = model,
            folds = 10)