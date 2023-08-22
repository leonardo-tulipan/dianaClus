# Structure

* data_prep.py: reads the data, extracts categorical and numerical
* main.py: autoencoder + umap + hdbscan, saves the encoder and the manifold
* prediction_step_n2d.py:  reads encoder and manifold(produced by main.py) to produce preditions on new data using HDBSCAN
* cldf.py: Uses default configuration to cluster data using two parameters, num_clus and num_epochs, python cldf.py num_clus num_epochs
