import pandas as pd
import numpy as np
from sklearn import model_selection 
import matplotlib.pyplot as plt           

fold = 5
df = pd.read_csv('train.csv')
#X = df.iloc[:,:-1]
#y = df.iloc[:,-1]
y_col = 'LST'

df["K-FOLD"] = -1   
df = df.sample(frac = 1).reset_index(drop=True)  

num_bins = int(np.floor(1 + np.log2(len(df))))  

df.loc[:, "BINS"] = pd.cut(  
        df[y_col], 
        bins = num_bins, 
        labels = False
        )  

kf = model_selection.StratifiedKFold(n_splits = fold)          

for f, (t_, v_) in enumerate(kf.split(X = df, 
                              y = df.BINS.values)):  

    df.loc[v_, 'K-FOLD'] = f
    train = df[df['K-FOLD'] != f]
    test = df[df['K-FOLD'] == f]
#    print(train['LST'].describe())
    train.boxplot(['BLUE'])
    plt.xlabel('train')
    plt.show()
    test.boxplot(['BLUE'])
    plt.xlabel('test')
    plt.show()
    print('----------')
