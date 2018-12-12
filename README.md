## Motif Extraction and Point-Pattern Matching

Additional information for the different parts of the repo can be found in the READMEs in specific directories.

### Code

This repository contains two pieces of software: Motif Extraction and Point-Pattern Matching. The code for these projects are in the `packages/` directory.

### Examples

The `examples/` directory contains examples for both projects.

### `voronoi.cc`

The `voronoi.cc` file can be compiled by linking to the Voro++ library, and the resulting executable should be named `voronoi` and placed on your system PATH. To compile the file by linking to Voro++, run this command (it may need some modification for your system): `g++ -Wall -ansi -pedantic -O3 -I../../src -L../../src -o voronoi voronoi.cc -lvoro++`

### Citations

How to Cite this Work:

...
