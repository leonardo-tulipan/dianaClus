import pandas as pd
import joblib
from config import *
from utils import extract_numerical, extract_categorical


with open(PATH + 'sufijos.txt') as file:
    names = file.readlines()


def gen_data(div):
    dfc0 = pd.read_csv(f"{PATH}ventasxsemanaclus_{names[-1].strip()}.csv")
    cols = [f"{i}_tot" for i in range(100) if f"{i}_tot" in dfc0.columns]
    cols += ['TOTAL_NETO_T','Momentum',
            'freqmes', 'dias_compra', 'recency_class', 'grupo_estrato','antiguedad', 'TIPOLOGIA', 'VENDEDOR_T']
    mask = dfc0['COD_CLIENTE']!= 0

    dfc0 = dfc0.loc[mask]
    if div:
        for col in cols:
            if '_tot' in col:
                dfc0[col] = dfc0[col]/dfc0['TOTAL_NETO_T']

    dfc = dfc0[cols].copy() 
    numerical = dfc.select_dtypes(include=["float", "int"])
    null_cols = []
    for col in numerical.columns:
        if sum(dfc[col]) == 0:
            null_cols.append(col)

    dfc.drop(columns=null_cols, inplace=True)
    dfc.dropna(inplace=True)
    original_index = dfc.reset_index(drop=False)['index']
    dfc.reset_index(inplace=True, drop=True)

    original_data = dfc0.loc[original_index]
    original_data.reset_index(inplace=True, drop=True)

    numerical = extract_numerical(dfc)
    categorical = extract_categorical(dfc)
    
    return original_data, numerical, categorical

def gen_data2():
    data = joblib.load('Data/data_emb2.pkl')
    return data
