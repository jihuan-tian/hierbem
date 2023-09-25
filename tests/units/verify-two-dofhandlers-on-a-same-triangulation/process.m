clear all;
load verify-two-dofhandlers-on-a-same-triangulation.output;

dof_handler1_vertex_indices - dof_handler2_vertex_indices
norm(dof_handler1_vertex_indices - dof_handler2_vertex_indices, 'fro')
