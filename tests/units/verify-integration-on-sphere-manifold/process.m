clear all;

load 'verify-integration-on-sphere-manifold.output';

sphere_area_true = 4 * pi;
relative_errors = (sphere_areas - sphere_area_true) ./ sphere_area_true;

mapping_num = size(sphere_areas, 1);
refinement_num = size(sphere_areas, 2) - 1;

legend_str = cell(mapping_num, 1);

figure;
hold on;
## Iterate over each mapping order.
for m = 1:mapping_num
  semilogy(0:refinement_num, abs(relative_errors(m, :)), 'linestyle', '-', 'color', clist(m, :));
  legend_str{m} = cstrcat("mapping order=" , num2str(m));
endfor
xlabel('Number of refinement');
ylabel('Relative error of sphere area');
grid on;
legend(legend_str, 'location', 'eastoutside');
scale_fig(gcf, [2.5, 2]);
PrintGCFLatex('relative-errors-vs-refinement.png');

legend_str = cell(refinement_num + 1, 1);

figure;
hold on;
## Iterate over each refinement.
for m = 0:refinement_num
  semilogy(1:mapping_num, abs(relative_errors(:, m+1)), 'linestyle', '-', 'color', clist(m+1, :));
  legend_str{m+1} = cstrcat("refine number=", num2str(m));
endfor
xlabel('Mapping order');
ylabel('Relative error of sphere area');
grid on;
legend(legend_str, 'location', 'eastoutside');
scale_fig(gcf, [2.5, 2]);
PrintGCFLatex('relative-errors-vs-mapping-order.png');
