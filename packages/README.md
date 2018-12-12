## PPM

The `ppm3d/_ppm3d/` directory contains the source code used to perform point-pattern matching [ref]. It is likely not worth reading if you are just trying to use this package. Instead, focus your attention on file `single_alignment.py` and the contents of `tipes/` (the latter defines the datastructures/classes that hold the data for PPM).

## Motif Extraction

This code uses PPM to perform all-to-all alignments of the clusters in an atomic model.

The `alignment/` directory performs all pairwise alignments of the clusters. The `clustering/` directory performs HDBSCAN to identify groups of similar clusters. The `analyze_results/` directory takes the results from `clustering/` and creates the motifs.
