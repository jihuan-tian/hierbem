clear all;
ConfigGraphicsToolkit;

addpath("/home/jihuan/Projects/git/octbem");
addpath("/home/jihuan/Projects/git/octbem/gmsh_io");

pkg load fpl;
pkg load msh;

mesh_filename = "./surface-jacobian-det-square.msh";
[path_name, file_name, file_ext] = fileparts(mesh_filename);
problem_domain_mesh = ReadGmshQuads(mesh_filename);

## ## Plot the mesh.
## figure;
## quadplot3d(problem_domain_mesh.mesh_cells, problem_domain_mesh.mesh_nodes, "color", "r", "linestyle", "-", "marker", ".");
## hold on;

## #############################################################################
## Generate the shape function space for describing the cell geometry, the order
## of which is determined by the mesher.
## #############################################################################
## At present, assume all cells in the mesh have the same geometric
## order, which is only 1st order at the moment.
shape_function_space_order = problem_domain_mesh.cell_geom_orders(2);
## Generate the series of shape functions dependent on the first two components
## of the area coordinate, while the last component is not independent.
shape_function_space = LagrangeBasisOn3DQuad(shape_function_space_order);
## Generate the local coordinates for the support nodes of shape functions.
shape_function_support_nodes = AreaCoordsOnQuad(shape_function_space_order);
## Total number of basis functions in the shape function space.
number_of_bases_in_shape_function_space = length(shape_function_space);

surface_jacobian_det_vector = zeros(problem_domain_mesh.number_of_cells, 1);
unit_cell_center = [0.5, 0.5];
for e = 1:problem_domain_mesh.number_of_cells
  J_functor = @(area_coord) GlobalSurfaceMetricOn3DQuad(area_coord, problem_domain_mesh.mesh_nodes(problem_domain_mesh.mesh_cells(e, :), :));
  surface_jacobian_det_vector(e) = J_functor(unit_cell_center);
  printf("%.5e\n", surface_jacobian_det_vector(e));
endfor

## Add a trailing newline to be consistent with the output of deal.ii.
printf("\n");
