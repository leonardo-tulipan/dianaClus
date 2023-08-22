from data_prep import *
import n2d 
import hdbscan
import umap
import wandb
import psutil
import os

def set_cpu_affinity(pid, cores):
    p = psutil.Process(pid)
    p.cpu_affinity(cores)

# Replace 'YOUR_PYTHON_PROCESS_PID' and 'CORES_LIST' accordingly
your_pid = os.getpid()
allowed_cores = [i for i in range(33)]  # For example, [0, 1] to run on cores 0 and 1

set_cpu_affinity(your_pid, allowed_cores)

config_ = {"latent_dim": 10,
          "min_samples": 7,
          "dataset": "1308",
          "epochs": 100,
          "n_neighbors": 30,
          }

wandb.login()
wandb.init(
            # set the wandb project where this run will be logged
            project="dian-clus-org",
                    
            # track hyperparameters and run metadata
            config=config_
            )




def approx_predict(n2d_obj, newdata):
    embedding = n2d_obj.encoder.predict(newdata)
    manifold = n2d_obj.manifolder.transorm(embedding)
    labs, probs = hdbscan.approximate_predict(n2d_obj.clusterer, manifold)
    return labs, probs

print('Numerical:', numerical.shape)
print('Categorical:', categorical.shape)
df_for_n2d = pd.concat([numerical, categorical * 1], axis=1)



X = df_for_n2d.values
latent_dim = config_['latent_dim']

# hdbscan arguments
hdbscan_args = {"min_samples":config_['min_samples'],"min_cluster_size":int(len(df_for_n2d)*.05), 'prediction_data':True,
        "gen_min_span_tree":True}

# umap arguments
umap_args = {"metric":"euclidean", "n_components":config_['latent_dim'], "n_neighbors":config_['n_neighbors'],"min_dist":0}

ae = n2d.AutoEncoder(input_dim = X.shape[-1], latent_dim = latent_dim) 

db_clust = n2d.manifold_cluster_generator(umap.UMAP, umap_args, hdbscan.HDBSCAN, hdbscan_args)
n2d_db = n2d.n2d(ae, db_clust)
embedding = n2d_db.fit(X, epochs = config_['epochs'])
n2d.save_n2d(n2d_db, 'Data/Models/enco.h5','Data/Models/gmm.sav')

# the probabilities
print(n2d_db.clusterer.probabilities_)
# the labels
print(n2d_db.clusterer.labels_)
# the relative validity
DBCV = n2d_db.clusterer.relative_validity_
print('DBCV', DBCV)

clustered = original_data.copy() 
clustered['LABELS'] =  n2d_db.clusterer.labels_

cluster_count = clustered['LABELS'].nunique()
clustered.to_csv(f"Data/Clustering_Outputs/clustered{clustered['LABELS'].nunique()}-{config_['dataset']}.csv")

noise_data = clustered['LABELS']==-1
noise = clustered['LABELS'].value_counts()[-1]/len(clustered)
wandb.log({"DBCV": DBCV, "cluster_count": cluster_count, "noise": noise })
wandb.finish()