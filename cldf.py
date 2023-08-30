from data_prep import *
from config import *
import sys
import n2d

import psutil
import os
def set_cpu_affinity(pid, cores):
    p = psutil.Process(pid)
    p.cpu_affinity(cores)
# Replace 'YOUR_PYTHON_PROCESS_PID' and 'CORES_LIST' accordingly
your_pid = os.getpid()
allowed_cores = [i for i in range(33)]  # For example, [0, 1] to run on cores 0 and 1
set_cpu_affinity(your_pid, allowed_cores)

df_for_n2d = pd.concat([numerical, categorical * 1], axis=1)
X = df_for_n2d.values
PATH_M = 'Data/Models/'
def run_cl(num_clus, num_epochs):
    latent_dim = num_clus
    ae = n2d.AutoEncoder(X.shape[-1], latent_dim)
    manifoldGMM = n2d.UmapGMM(num_clus, umap_metric="precomputed")
    cluster = n2d.n2d(ae, manifoldGMM)
    cluster.fit(X, epochs = num_epochs)
    labels = cluster.predict(X)
    n2d.save_n2d(cluster, f"{PATH_M}latest-enco.h5", f"{PATH_M}gmm-latest.sav")
    clustered = original_data.copy()
    clustered['LABELS'] = labels
    clustered.to_csv(f"Data/Clustering_Outputs/clustered{num_clus}-cld{num_epochs}-2908.csv")

if __name__ == '__main__':
    n_clus = int(sys.argv[1])
    n_epochs = int(sys.argv[2])
    run_cl(n_clus, n_epochs)



