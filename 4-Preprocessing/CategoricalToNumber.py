import pandas as pd

df = pd.read_csv("test_categorical.csv", encoding='unicode_escape')

def label_encode(df, colnames):
    for colname in colnames:
        labels = df[colname].unique()
        for index, label in enumerate(labels):
            df.loc[df[colname] == label, colname] = index

print(df, end='\n\n')
label_encode(df, ['Renk Tercihi', 'Cinsiyet'])
print(df, end='\n\n')
