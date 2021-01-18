from skimage.filters import threshold_otsu
import pandas as pd
import os
def binarize(df,
            column_name):

    misc_output_path = os.path.join(os.getcwd(), 
                    'output_rs_learn',
                    'misc')     

    if not os.path.exists(misc_output_path):
        os.makedirs(misc_output_path)

    df = df[column_name]
    thresh = threshold_otsu(df)
    binary = df > thresh
    binary = binary.astype(int)
    df_binary = pd.DataFrame(binary).set_index(df.index.values)
    print(f'Theshold: {thresh}')

    df_binary.to_csv(os.path.join(misc_output_path,
     'binary_%s.csv'%column_name),
            index = False)

    print('Binarization done for %s'%column_name)
    return df_binary