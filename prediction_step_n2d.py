import n2d
import umap
import hdbscan
import pickle
from data_prep import *
from config import *
from keras.models import load_model
PATH_M = "Data/Models/"

df_for_n2d = pd.concat([numerical, categorical * 1], axis=1)
X = df_for_n2d.values

def approx_predict(manifold_name, encoder_name, newdata):
    man = pickle.load(open(f"{PATH_M}{manifold_name}",'rb')) 
    umap_obj = man.manifold_in_embedding
    encoder = load_model(f"{PATH_M}{encoder_name}")
    embedding = encoder.predict(newdata)
    manifold = umap_obj.transform(embedding)
    labs, probs = hdbscan.approximate_predict(man.cluster_manifold, manifold)
    return labs, probs


approx = approx_predict('gmm.sav','enco.h5', X[:10])
print(approx)



