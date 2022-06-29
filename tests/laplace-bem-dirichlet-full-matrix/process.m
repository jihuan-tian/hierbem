clear all;
load laplace-bem-dirichlet-full-matrix.output;

## Generate the grid points for potential evaluation.
nx = 100;
ny = 100;
xdim = 4.0;
ydim = 4.0;
xstart = -xdim / 2;
ystart=  -ydim / 2;
xend = xstart+xdim;
yend = ystart+ydim;
dx = xdim / nx;
dy = ydim / ny;

## Generate the regular grid.
[xx, yy] = meshgrid(xstart:dx:xend, ystart:dy:yend);
## Linearize the grid coordinate components. In the arranged grid points, the x
## coordinate components run faster than y coordinate components.
x = reshape(xx, numel(xx), 1);
y = reshape(yy, numel(yy), 1);
z = 4.0 * ones(length(x), 1);

zz = griddata(x, y, potential_values, xx, yy, "linear");

## Calculate the analytical solution.
x0 = [0.25, 0.25, 0.25];
potential_func = @(x,y,z) 1.0/4.0/pi./sqrt((x-x0(1)).^2 + (y-x0(2)).^2 + (z-x0(3)).^2);
potential_values_analytical = potential_func(x, y, z);
zz_analytical = griddata(x, y, potential_values_analytical, xx, yy, "linear");

printout_var("norm(potential_values - potential_values_analytical, 'fro') / norm(potential_values_analytical, 'fro')");
printout_var("norm(potential_values - potential_values_analytical, 'Inf') / norm(potential_values_analytical, 'Inf')");
printout_var("norm(potential_values - potential_values_analytical, 2) / norm(potential_values_analytical, 2)");
printout_var("norm(potential_values - potential_values_analytical, 1) / norm(potential_values_analytical, 1)");

figure;
subplot(1, 2, 1);
surfc(xx, yy, zz, "edgecolor", "none");
title("Numerical solution");
subplot(1, 2, 2);
surfc(xx, yy, zz_analytical, "edgecolor", "none");
title("Analytical solution");

figure;
subplot(1, 2, 1);
surfc(xx, yy, zz, "edgecolor", "none");
title("Numerical solution");
axis equal;
axis tight;
view(2);
subplot(1, 2, 2);
surfc(xx, yy, zz_analytical, "edgecolor", "none");
title("Analytical solution");
axis equal;
axis tight;
view(2);

figure;
plot(potential_values, "r-o");
hold on;
plot(potential_values_analytical, "b-+");
legend({"Numerical solution", "Analytical solution"});
title("Potential distribution on a plane");
