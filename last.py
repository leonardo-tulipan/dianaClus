import pandas as pd
import sys
def retrieve_last(path, name):
    df = pd.read_csv(path+name)
    gr = df.groupby('NOMBRE_CLIENTE').last()
    gr.to_csv(path+'last'+name)

if __name__=='__main__':
    path = sys.argv[1]
    name = sys.argv[2]
    retrieve_last(path, name)


