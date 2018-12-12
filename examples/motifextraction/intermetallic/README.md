### Intermetallic Example

We provide a CuZr2 intermetallic model in XYZ format (see `modelfiles/`) and the motifs of this model are identified. The workflow is presented below.

0) First we describe the directory layout. The `data/` directory stores all the data output by PPM and ME. The `alignment/` directory performs all pairwise alignments of the clusters. The `clustering/` directory performs HDBSCAN to identify groups of similar clusters. The `analyze_results/` directory takes the results from `clustering/` and creates the motifs. The `motif_analysis/` directory provides a Jupyter notebook with cursory analysis.

1) The nearest neighbors of each atom in the model are identified. We crudely ignore "surface" atoms (because the model is in XYZ rather than CIF format and the non-cubic PBCs are not defined in this code) by only considering atoms with 10+ neighbors. The center atom + nearest neighbor structure (aka "clusters") are identified and extracted by running `python generate_clusters_for_alignment.py modelfiles/CuZr2_324.xyz` from the current directory directory. The clusters are saved in `data/clusters/`.

2) The all-to-all alignment of the clusters is performed by running `./setup.sh`. This file primarily runs `alignment/align_clusters.py` with each cluster as input; this file aligns all of the clusters in `data/clusters/` to the input structure. The results are stored in `data/results/`. Four metrics are calculated for each alignment using the file `alignment/extract_errors.py`; the results are stored in `data/errors/`.

3) The error metrics are then reformatted into a set of dissimilarity matrices, and the geometric mean of the matrices is calculated to create a final dissimilarity matrix that we provide to HDBSCAN for clustering. This is done by running `python create_affinities.py` from the `clustering/` directory.

4) The `clustering/clustering.py` file performs HDBSCAN. Please see this file as well as the package file `packages/motifextraction/clustering/clustering.py` and the HDBSCAN documentation for fine tuning of the paramters. However, hopefully the only parameter you will need to tune is `min_cluster_size_for_hdbscan` in `./clustering/clustering.py`. See the contents of `clustering/run.sh` for a succinct version of this workflow.

5) The clustering results of HDBSCAN save the indices of similar clusters in separate files (see `data/affinities/...`). It's useful to rename these files in some circumstances, and running `python refine_labels.py ../data/affinities/combined_affinity.npy` does this.

6) Now we can create the motifs from the clustered clusters via `python create_motifs.py ../data/affinities/` from the `analyze_results/` directory. Finally, we will align each cluster to each motif and calculate the corresponding similarities (see `analyze_results/run_locally.sh`).

7) A cursory analysis of the motifs is provided in the Jupyer notebook `motif_analysis/motifs.ipynb`. Rather than repeat the results/conclusions here, it's more informative to read the notebook.
