import pandas as pd
import numpy as np

def delete_na(df):
    return df.dropna(axis=0)

def fill_na(df,method=0):
    '''
    fill nan value with:
    method=1:0 default
    method=2:upper value 
    method=3:behand value
    '''
    if method==0:
        return df.fillna(0)
    elif method==1:
        return df.fillna(method='pad')
    elif method==2:
        return df.fillna(method='bfill')

def drop_na(df,method=0):
    df = df.dropna()


def fill_na_with_colmean(df,colname):
    '''
    fill column's nontype value with mean of each group which is 
    grouped by colname
    '''
    groups = df[colname].unique()
    ndf = pd.DataFrame()
    for ind in groups:
        t = df[colname] == ind
        a = df[t].fillna(df[t].mean())
        ndf = ndf.append(a)
    return ndf

df = pd.DataFrame({'code':[1,2,3,4,5,6,7,8],
                   'value':[np.nan,5,7,8,9,10,11,12],                   
                   'value2':[5,np.nan,7,np.nan,9,10,11,12],
                   'indstry':['农业1','农业1','农业1','农业2','农业2','农业4','农业2','农业3']},
                    columns=['code','value','value2','indstry'],
                    index=list('ABCDEFGH'))

print(df)
print(fill_na_with_groupmean(df,'indstry'))