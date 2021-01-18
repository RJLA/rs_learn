from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import pandas as pd

def fix_imbalance(X_train,
                  y_train):

    over = SMOTE()
    under = RandomUnderSampler()
    
    steps = [('o', over), 
             ('u', under)]
    
    pipeline = Pipeline(steps = steps)
    
    X_resampled, y_resampled = pipeline.fit_resample(X_train, 
                                 y_train)
#    
    X_res_df = pd.DataFrame(X_resampled,
                            columns = X_train.columns).reset_index(drop = True)
    y_res_df = pd.DataFrame(y_resampled).reset_index(drop = True)
#    
    return X_res_df, y_res_df

#import pandas as pd
#from sklearn.preprocessing import LabelEncoder
#
#df = pd.read_csv(r"C:\Users\Dlaniger\Anaconda3\Lib\site-packages\rs_learn\sample_dataframe.csv")
#X, y = df.iloc[:,:-2], df.iloc[:,-1]
#lec = LabelEncoder()
#y_enc = lec.fit_transform(y)
#
#X_resampled, y_resampled = fix_imbalance(X,y_enc)
    # summarize the new class distribution
    # scatter plot of examples by class label
