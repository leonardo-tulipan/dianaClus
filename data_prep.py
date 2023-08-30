import pandas as pd

from config import PATH
from utils import extract_numerical, extract_categorical

dfc0 = pd.read_csv(f"{PATH}ventasxsemanaclus2908.csv")
cols = [f"{i}_tot" for i in range(100) if f"{i}_tot" in dfc0.columns]
cols += ['TOTAL_NETO_T','Momentum',
         'freqmes', 'dias_compra', 'recency_class', 'estrato']

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


