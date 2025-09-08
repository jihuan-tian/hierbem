## Copyright (C) 2021-2023 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

clear all;
ConfigGraphicsToolkit;

addpath("~/Projects/git/octbem/gmsh_io");
addpath("~/Projects/git/octbem/stroud");

pkg load fpl;
pkg load msh;

mesh_filename = "./six-cells-for-octave.msh";
[path_name, file_name, file_ext] = fileparts(mesh_filename);
problem_domain_mesh = ReadGmshQuads(mesh_filename);

## #############################################################################
## Generate the shape function space for describing the cell geometry, the order
## of which is determined by the mesher.
## #############################################################################
## At present, assume all cells in the mesh have the same geometric
## order, which is only 1st order at the moment.
shape_function_space_order = problem_domain_mesh.cell_geom_orders(1);
## Generate the series of shape functions dependent on the first two components
## of the area coordinate, while the last component is not independent.
shape_function_space = LagrangeBasisOn3DQuad(shape_function_space_order);
## Generate the local coordinates for the support nodes of shape functions.
shape_function_support_nodes = AreaCoordsOnQuad(shape_function_space_order);
## Total number of basis functions in the shape function space.
number_of_bases_in_shape_function_space = length(shape_function_space);

## #############################################################################
## Generate the test and ansatz(trial) function spaces: the Lagrange functions
## are used.
## #############################################################################
## At present, the code for generating high order nodes from the lower nodes
## (directly obtained from the mesher) is not implemented. Therefore, the test
## function space order is the same as the shape function space order, which is
## constructed directly from the mesher's output.
## 
## N.B. The highest order should not be larger than 2 for comparison
## with deal.ii. Because deal.ii adopts Gauss-Lobatto support points,
## which are different from the equi-distant support points in octbem.
test_function_space_order = 2;
test_function_space = LagrangeBasisOn3DQuad(test_function_space_order);
## Generate the support nodes associated with basis functions in the test
## function space, keeping the same sequence as the basis functions.
test_function_support_nodes = AreaCoordsOnQuad(test_function_space_order);
## Total number of basis functions in the test function space.
number_of_bases_in_test_function_space = length(test_function_space);

## In Galerkin method, the ansatz(trial) function space is selected to be the
## same as test function space.
ansatz_function_space_order = test_function_space_order;
ansatz_function_space = test_function_space;
ansatz_function_support_nodes = test_function_support_nodes;
number_of_bases_in_ansatz_function_space = length(ansatz_function_space);

## Generate 4d Gauss-Legendre quadrature points and weights for
## Sauter's method.
norder_for_same_panel = 4;
norder_for_common_edge = 3;
norder_for_common_vertex = 3;
norder_for_regular = 2;

global sauter_same_panel_4d_qpts sauter_same_panel_4d_qwts sauter_common_edge_4d_qpts sauter_common_edge_4d_qwts sauter_common_vertex_4d_qpts sauter_common_vertex_4d_qwts sauter_regular_4d_qpts sauter_regular_4d_qwts;

[qpts_1d, qwts_1d] = GaussLegendreRule(norder_for_same_panel);
## Adjust the Gauss-Legendre rule from [-1, 1] to [0, 1].
[qpts_1d, qwts_1d] = rule_adjust(-1, 1, 0, 1, norder_for_same_panel, qpts_1d, qwts_1d);
sauter_same_panel_4d_qpts = tensor_prod4(qpts_1d, qpts_1d, qpts_1d, qpts_1d);
sauter_same_panel_4d_qwts = tensor_prod4(qwts_1d, qwts_1d, qwts_1d, qwts_1d);

[qpts_1d, qwts_1d] = GaussLegendreRule(norder_for_common_edge);
## Adjust the Gauss-Legendre rule from [-1, 1] to [0, 1].
[qpts_1d, qwts_1d] = rule_adjust(-1, 1, 0, 1, norder_for_common_edge, qpts_1d, qwts_1d);
sauter_common_edge_4d_qpts = tensor_prod4(qpts_1d, qpts_1d, qpts_1d, qpts_1d);
sauter_common_edge_4d_qwts = tensor_prod4(qwts_1d, qwts_1d, qwts_1d, qwts_1d);

[qpts_1d, qwts_1d] = GaussLegendreRule(norder_for_common_vertex);
## Adjust the Gauss-Legendre rule from [-1, 1] to [0, 1].
[qpts_1d, qwts_1d] = rule_adjust(-1, 1, 0, 1, norder_for_common_vertex, qpts_1d, qwts_1d);
sauter_common_vertex_4d_qpts = tensor_prod4(qpts_1d, qpts_1d, qpts_1d, qpts_1d);
sauter_common_vertex_4d_qwts = tensor_prod4(qwts_1d, qwts_1d, qwts_1d, qwts_1d);

[qpts_1d, qwts_1d] = GaussLegendreRule(norder_for_regular);
## Adjust the Gauss-Legendre rule from [-1, 1] to [0, 1].
[qpts_1d, qwts_1d] = rule_adjust(-1, 1, 0, 1, norder_for_regular, qpts_1d, qwts_1d);
sauter_regular_4d_qpts = tensor_prod4(qpts_1d, qpts_1d, qpts_1d, qpts_1d);
sauter_regular_4d_qwts = tensor_prod4(qwts_1d, qwts_1d, qwts_1d, qwts_1d);

## 2D quadrature rule for normal integration on quadrangles.
## Quadrature norder (number of quadrature points) for the integrals in FEM,
## which will also be used in BEM. N.B. n-point Gauss quadrature is exact for
## polynomials of degree \f$2n - 1\f$.
## Ref: https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Legendre_quadrature
fem_gauss_quad_norder_2d = ceil((test_function_space_order + ansatz_function_space_order + 1) / 2);
## Generate the Gauss-Legendre quadrature rules to perform 2D FEM quadrature.
[fem_gauss_quad_2d_qpts_xi1, fem_gauss_quad_2d_qpts_xi2, fem_gauss_quad_2d_qwts] = triangle_unit_product_set(fem_gauss_quad_norder_2d);
fem_gauss_quad_2d_qpts = [fem_gauss_quad_2d_qpts_xi1', fem_gauss_quad_2d_qpts_xi2'];
fem_gauss_quad_2d_qwts = fem_gauss_quad_2d_qwts';

cell_matrix_slp = zeros(number_of_bases_in_test_function_space, number_of_bases_in_ansatz_function_space);
cell_matrix_dlp = zeros(number_of_bases_in_test_function_space, number_of_bases_in_ansatz_function_space);
cell_matrix_adlp = zeros(number_of_bases_in_test_function_space, number_of_bases_in_ansatz_function_space);
cell_matrix_hyper = zeros(number_of_bases_in_test_function_space, number_of_bases_in_ansatz_function_space);

## Calculate the distance between each pair of panels.
panel_distance_matrix = CalcQuadPanelDistanceMatrix(problem_domain_mesh);

## Calculate the neighboring type between each pair of panels.
neighboring_type_matrix = CalcQuadPanelNeighboringTypes(problem_domain_mesh.mesh_cells);

e = 1;
f = 6;

Jx = @(kx_area_coord) GlobalSurfaceMetricOn3DQuad(kx_area_coord, problem_domain_mesh.mesh_nodes(problem_domain_mesh.mesh_cells(e, :), :));
Jy = @(ky_area_coord) GlobalSurfaceMetricOn3DQuad(ky_area_coord, problem_domain_mesh.mesh_nodes(problem_domain_mesh.mesh_cells(f, :), :));

fprintf(stdout(), "Calculating SLP...");
for i = 1:number_of_bases_in_test_function_space
  for j = 1:number_of_bases_in_ansatz_function_space
    cell_matrix_slp(i, j) = SauterQuadRuleFlat(@LaplaceSLPKernel3D, test_function_space{i}, ansatz_function_space{j}, shape_function_space, shape_function_space, e, f, panel_distance_matrix, neighboring_type_matrix, problem_domain_mesh.mesh_cells, problem_domain_mesh.mesh_nodes, problem_domain_mesh.cell_normal_vectors(e, :), problem_domain_mesh.cell_normal_vectors(f, :), Jx, Jy, problem_domain_mesh.max_cell_range);
  endfor
endfor

fprintf(stdout(), "Calculating DLP...");
for i = 1:number_of_bases_in_test_function_space
  for j = 1:number_of_bases_in_ansatz_function_space
    cell_matrix_dlp(i, j) = SauterQuadRuleFlat(@LaplaceDLPKernel3DFlat, test_function_space{i}, ansatz_function_space{j}, shape_function_space, shape_function_space, e, f, panel_distance_matrix, neighboring_type_matrix, problem_domain_mesh.mesh_cells, problem_domain_mesh.mesh_nodes, problem_domain_mesh.cell_normal_vectors(e, :), problem_domain_mesh.cell_normal_vectors(f, :), Jx, Jy, problem_domain_mesh.max_cell_range);
  endfor
endfor

fprintf(stdout(), "Calculating ADLP...");
for i = 1:number_of_bases_in_test_function_space
  for j = 1:number_of_bases_in_ansatz_function_space
    cell_matrix_adlp(i, j) = SauterQuadRuleFlat(@LaplaceDLPAdjointKernel3DFlat, test_function_space{i}, ansatz_function_space{j}, shape_function_space, shape_function_space, e, f, panel_distance_matrix, neighboring_type_matrix, problem_domain_mesh.mesh_cells, problem_domain_mesh.mesh_nodes, problem_domain_mesh.cell_normal_vectors(e, :), problem_domain_mesh.cell_normal_vectors(f, :), Jx, Jy, problem_domain_mesh.max_cell_range);
  endfor
endfor

fprintf(stdout(), "Calculating Hyper...");
for i = 1:number_of_bases_in_test_function_space
  for j = 1:number_of_bases_in_ansatz_function_space
    cell_matrix_hyper(i, j) = SauterQuadRuleFlat(@LaplaceHyperSingKernel3DFlat, test_function_space{i}, ansatz_function_space{j}, shape_function_space, shape_function_space, e, f, panel_distance_matrix, neighboring_type_matrix, problem_domain_mesh.mesh_cells, problem_domain_mesh.mesh_nodes, problem_domain_mesh.cell_normal_vectors(e, :), problem_domain_mesh.cell_normal_vectors(f, :), Jx, Jy, problem_domain_mesh.max_cell_range);
  endfor
endfor

save("-binary", "sauter_quad_regular.bin");
