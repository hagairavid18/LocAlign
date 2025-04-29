# Correspondence denoiser
## Goal

Goal of the module: reinforce correspondences that are mutually consistent, and dampen correspondences that are inconsistent.


## Inputs:
- Soft Correspondence matrix (output of the BBR) $C_{ij}$
- Frames of the src atoms $F_{ik}$
- Frames of the target atoms $F_{jk}$

Hyperparameter: K, maximum number of correspondences.


## Sketch:

- Pick top K correspondences out of the matrix.

- Build a graph where each node is a correspondence and there are edges between every pair of correspondences.

- For now, the node feature is a scalar. Its initial value is the value of en 

Its initial value with 

- For now, the node and edge features are both scalar.

The node values equal the correspondence matrix value, and the edge matrix 

- Apply a GNN update.
- The edge value should be a function