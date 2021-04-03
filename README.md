# Motif Extraction and Point-Pattern Matching (3D)

This code uses three-dimensional point-pattern matching (ppm3d) to perform all-to-all alignments of the clusters in an atomic model. See the following papers for more information:
https://arxiv.org/pdf/1901.04124.pdf
https://arxiv.org/pdf/1811.06098.pdf
https://arxiv.org/pdf/1901.07014.pdf

The latest version of the PPM code can be found at: https://github.com/spatala/ppm3d

## Examples

The `examples/` directory contains examples for both projects.

Inside the `packages/motifextraction` directory you will find three relevant directories. The `alignment/` directory performs all pairwise alignments of the clusters. The `clustering/` directory performs HDBSCAN to identify groups of similar clusters. The `analyze_results/` directory takes the results from `clustering/` and creates the motifs.

The `packages/ppm3d` directory contains the ppm3d code that was modified from https://github.com/spatala/ppm3d that enables the motif extraction workflow. The code in their repo will need to be downloaded and compiled as well (see the installation instructions below).

## Citations

Please cite any of the relevant papers below (you can find the published DOIs):

* https://arxiv.org/pdf/1901.04124.pdf (Published here: https://www.sciencedirect.com/science/article/abs/pii/S1359645419302721)
* https://arxiv.org/pdf/1811.06098.pdf
* https://arxiv.org/pdf/1901.07014.pdf

See https://github.com/spatala/ppm3d for information on how to cite the ppm3d package written by Arash Dehghan Banadaki and Srikanth Patala.


## Installation

In order to use Motif Extraction, you need to install two packages: `motifextraction` and `ppm3d`, both located in this repository. The `ppm3d` package in this repo relies heavily on the point-pattern matching code written by Arash Dehghan Banadaki and Srikanth Patala, which can be found here: https://github.com/spatala/ppm3d

First, download or `git clone` this repository to your computer.

### Installing Point-Pattern Matching 3D

After you have downloaded this repo, open a terminal and `cd` to the `.../motifextraction/packages/ppm3d/` directory. Then download or `git clone` Banadaki and Patala's ppm3d code from here: https://github.com/spatala/ppm3d  Once downloaded, change the name of the downloaded folder from `ppm3d` to `_ppm3d`. `cd` into `_ppm3d`, then install the code following their installation instructions. It may take some time to compile.

Modify your `PYTHONPATH` environment variable to include the directory: `.../motifextraction/packages/ppm3d`. On Linux/Mac this can be done by running `export PYTHONPATH=$PYTHONPATH:.../motifextraction/packages/ppm3d`.

### Installing Motif Extraction

To install this code, simply add it to your `PYTHONPATH` environment variable. Modify your `PYTHONPATH` environment variable to include the directory: `.../motifextraction/packages/motifextraction`. On Linux/Mac this can be done by running `export PYTHONPATH=$PYTHONPATH:.../motifextraction/packages/motifextraction`.

This code uses PPM to perform all-to-all alignments of the clusters in an atomic model.

### Creating the `voronoi` Executable

The `voronoi.cc` file can be compiled by linking to the Voro++ library, and the resulting executable should be named `voronoi` and placed on your `PATH` environment variable. To compile the file by linking to Voro++, run this command (it may need some modification for your system): `g++ -Wall -ansi -pedantic -O3 -I../../src -L../../src -o voronoi voronoi.cc -lvoro++`

