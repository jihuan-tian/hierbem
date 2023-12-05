clear all;

load_packages;
load U.dat;
load b.dat;
load hmatrix-transpose-forward-substitution.output;

y = forward_substitution(U', b);
norm(x - y, 'fro') / norm(y, 'fro')

figure;
hold on;
plot(x, 'b-.');
plot(y, 'r-.');

figure;
plot_bct_struct("H_bct.dat");
