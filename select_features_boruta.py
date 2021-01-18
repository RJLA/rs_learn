import pandas as pd
from boruta import BorutaPy
def select_features_boruta(X,
                    y,
                    model):      
    
    feat_selector = BorutaPy(model,
                             verbose = 2)
    
    feat_selector.fit(X.values, y)
     
    print('Number of original features: %s' %X.shape[1])
    print('Number of selected features: %s'%feat_selector.n_features_)

    X_filtered = feat_selector.transform(X.values)
    result = [x for x, y in zip(list(X.columns), 
                                list(feat_selector.support_)) if y == True]

    df_features_final = pd.DataFrame(X_filtered, 
                                     columns = result).set_index(X.index.values)
    print('Important features selected: %s'%result)
    return df_features_final 
