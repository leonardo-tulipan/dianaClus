from data_prep import *
import n2d 
import hdbscan
import umap
import umap.plot
import wandb
import psutil
import os
import sys
import matplotlib.pyplot as plt

projects = {'Diana': 'dian-clus-org',
            'Kikes': 'kikes-clus_emb'}

data_sets= {'Diana': names[-1].strip(),
            'Kikes': 'Q2'}

def set_cpu_affinity(pid, cores):
    p = psutil.Process(pid)
    p.cpu_affinity(cores)


def run(conf, kind):
    ref = f"-{conf['dataset']}-ld{conf['latent_dim']}-e{conf['epochs']}-ms{conf['min_samples']}-rescale-{conf['rescale']}-div{conf['div']}-min_cs{conf['min_cluster_size']}-"
    wandb.login()
    wandb.init(
            # set the wandb project where this run will be logged
            project= projects[kind],
                    
            # track hyperparameters and run metadata
            config=conf
            )
    
    if kind == 'Diana':
        original_data, numerical, categorical = gen_data(conf['div']==1)
        print('Numerical:', numerical.shape)
        print('Categorical:', categorical.shape)
        df_for_n2d = pd.concat([numerical, categorical * 1], axis=1)
    
        if conf['rescale'] == 1:
            df_for_n2d['TOTAL_NETO_T'] *= 20
        X = df_for_n2d.values
    else:
        X = gen_data2()
    latent_dim = config_['latent_dim']

    # hdbscan arguments
    hdbscan_args = {"min_samples":conf['min_samples'],"min_cluster_size":int(len(X)*conf['min_cluster_size']/100), 'prediction_data':True,
        "gen_min_span_tree":True}

    # umap arguments
    umap_args = {"metric":"euclidean", "n_components":conf['latent_dim'], "n_neighbors":conf['n_neighbors'],"min_dist":0, "random_state":SEED}

    ae = n2d.AutoEncoder(input_dim = X.shape[-1], latent_dim = latent_dim) 

    db_clust = n2d.manifold_cluster_generator(umap.UMAP, umap_args, hdbscan.HDBSCAN, hdbscan_args)
    n2d_db = n2d.n2d(ae, db_clust)
    embedding = n2d_db.fit(X, epochs = config_['epochs'])
    n2d.save_n2d(n2d_db, f"{PATH_N2D}Data/Models/enco{ref}-{kind}.h5",f"{PATH_N2D}Data/Models/gmm{ref}-{kind}.sav")

    # the probabilities
    print(n2d_db.clusterer.probabilities_)
    # the labels
    print(n2d_db.clusterer.labels_)
    # the relative validity
    DBCV = n2d_db.clusterer.relative_validity_
    print('DBCV', DBCV)
    if kind == 'Diana':
        clustered = original_data.copy() 
    else:
        clustered = pd.DataFrame(X) 
    clustered['LABELS'] =  n2d_db.clusterer.labels_

    cluster_count = clustered['LABELS'].nunique()
    clustered.to_csv(f"{PATH_N2D}Data/Clustering_Outputs/kike-clustered{clustered['LABELS'].nunique()}{ref}-{kind}.csv")

    noise_data = clustered['LABELS']==-1
    noise = clustered['LABELS'].value_counts()[-1]/len(clustered)
    wandb.log({"DBCV": DBCV, "cluster_count": cluster_count, "noise": noise })
    wandb.finish()
    return n2d_db, ref

if __name__ == '__main__':
    config_ = {"latent_dim": 30,
          "min_samples": int(sys.argv[1]),
          "min_cluster_size": float(sys.argv[2]),
          "dataset":data_sets[sys.argv[7]],
          "epochs": int(sys.argv[3]),
          "n_neighbors": int(sys.argv[4]),
          "rescale": int(sys.argv[5]),
          "div": int(sys.argv[6]),
          }
    # Replace 'YOUR_PYTHON_PROCESS_PID' and 'CORES_LIST' accordingly
    your_pid = os.getpid()
    allowed_cores = [i for i in range(33)]  # For example, [0, 1] to run on cores 0 and 1

    set_cpu_affinity(your_pid, allowed_cores)


    n2d_db, ref = run(config_, sys.argv[7])

    embedding = n2d_db.manifold_learner.hle
    out_embb = pd.DataFrame(embedding)
    labels = n2d_db.clusterer.labels_
    out_embb['LABELS'] = labels
    out_embb.to_csv(f"{PATH_N2D}Data/Clustering_Outputs/embedding{ref}-{sys.argv[7]}.csv")
    # Filter rows based on labels > -1
    mask = labels > -1

    # Apply the mask to both arrays
    filtered_embeddings = embedding[mask]
    filtered_labels = labels[mask]
    reducer = umap.UMAP(n_components=2)
    
    _2demb = reducer.fit_transform(filtered_embeddings)
    plot = umap.plot.points(reducer, labels = filtered_labels, theme='fire')

    plt.savefig(f'{PATH_N2D}Data/Clustering_Outputs/umap_plot{ref}-{sys.argv[7]}-{sys.argv[7]}.png', dpi=300, bbox_inches='tight') 


