clear all;

load_packages;
load L.dat;
load b.dat;
load hmatrix-block-triangular-forward-substitution-unit-diag.output;

y = L \ b;
norm(x - y, 'fro') / norm(y, 'fro')

figure;
hold on;
plot(x, 'b-.');
plot(y, 'r-.');

figure;
plot_bct_struct("H_bct.dat");
