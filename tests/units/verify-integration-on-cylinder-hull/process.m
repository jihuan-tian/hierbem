## Copyright (C) 2023 Jihuan Tian <jihuan_tian@hotmail.com>
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
load 'verify-integration-on-cylinder-hull.output';

cylinder_area_true = 2 * pi * 2;

## Relative errors using MappingQ with different orders.
relative_errors_of_mappingq = (cylinder_areas_mappingq - cylinder_area_true) / cylinder_area_true;
## Relative errors using MappingManifold under different quadrature point number in one direction.
relative_errors_of_mapping_manifold = (cylinder_areas_mapping_manifold - cylinder_area_true) / cylinder_area_true;

mapping_num = size(cylinder_areas_mappingq, 1);
refinement_num = size(cylinder_areas_mappingq, 2) - 1;
quad_num = size(cylinder_areas_mapping_manifold, 1);

legend_str = cell(mapping_num, 1);

figure;
hold on;
## Iterate over each mapping order.
for m = 1:mapping_num
  semilogy(0:refinement_num, abs(relative_errors_of_mappingq(m, :)), 'linestyle', '-', 'color', clist(m, :));
  legend_str{m} = cstrcat("mapping order=" , num2str(m));
endfor
xlabel('Number of refinement');
ylabel('Relative error of cylinder hull area');
title('Use MappingQ');
grid on;
legend(legend_str, 'location', 'eastoutside');
scale_fig(gcf, [2.5, 2]);
PrintGCFLatex('mappingq-relative-errors-vs-refinement.png');

legend_str = cell(refinement_num + 1, 1);

figure;
hold on;
## Iterate over each refinement.
for m = 0:refinement_num
  semilogy(1:mapping_num, abs(relative_errors_of_mappingq(:, m+1)), 'linestyle', '-', 'color', clist(m+1, :));
  legend_str{m+1} = cstrcat("refine number=", num2str(m));
endfor
xlabel('Mapping order');
ylabel('Relative error of cylinder hull area');
title('Use MappingQ');
grid on;
legend(legend_str, 'location', 'eastoutside');
scale_fig(gcf, [2.5, 2]);
PrintGCFLatex('mappingq-relative-errors-vs-mapping-order.png');

legend_str = cell(quad_num, 1);

figure;
hold on;
## Iterate over each quadrature number in one direction.
for m = 1:quad_num
  semilogy(0:refinement_num, abs(relative_errors_of_mapping_manifold(m, :)), 'linestyle', '-', 'color', clist(m, :));
  legend_str{m} = cstrcat("quad point number=" , num2str(m));
endfor
xlabel('Number of refinement');
ylabel('Relative error of cylinder hull area');
title('Use MappingManifold');
grid on;
legend(legend_str, 'location', 'eastoutside');
scale_fig(gcf, [2.5, 2]);
PrintGCFLatex('mappingmanifold-relative-errors-vs-refinement.png');

legend_str = cell(refinement_num + 1, 1);

figure;
hold on;
## Iterate over each refinement.
for m = 0:refinement_num
  semilogy(1:quad_num, abs(relative_errors_of_mapping_manifold(:, m+1)), 'linestyle', '-', 'color', clist(m+1, :));
  legend_str{m+1} = cstrcat("refine number=", num2str(m));
endfor
xlabel('Quadrature point number in one direction');
ylabel('Relative error of cylinder hull area');
title('Use MappingManifold');
grid on;
legend(legend_str, 'location', 'eastoutside');
scale_fig(gcf, [2.5, 2]);
PrintGCFLatex('mappingmanifold-relative-errors-vs-quad-num.png');
