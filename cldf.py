from data_prep import *
from config import *
import sys
import n2d

df_for_n2d = pd.concat([numerical, categorical * 1], axis=1)
X = df_for_n2d.values
PATH_M = 'Data/Models/'
def run_cl(num_clus, num_epochs):
    latent_dim = num_clus
    ae = n2d.AutoEncoder(X.shape[-1], latent_dim)
    manifoldGMM = n2d.UmapGMM(num_clus)
    cluster = n2d.n2d(ae, manifoldGMM)
    cluster.fit(X, epochs = num_epochs)
    n2d.save_n2d(cluster, f"{PATH_M}enco-cld.h5", f"{PATH_M}gmm-cld.sav")
    clustered = original_data.copy()
    clusterd['LABELS'] = cluster.clusterer.labels_
    clustered.to_csv(f"Data/Clustering_Outputs/clustered{num_clus}-cld{num_epochs}.csv")

if __name__ == '__main__':
    n_clus = int(sys.argv[1])
    n_epochs = int(sys.argv[2])
    run_cl(n_clus, n_epochs)



