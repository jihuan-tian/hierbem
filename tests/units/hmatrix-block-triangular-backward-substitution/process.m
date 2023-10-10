clear all;
load U.dat;
load b.dat;
load hmatrix-block-triangular-backward-substitution.output;

y = U \ b;
norm(x - y, 'fro') / norm(y, 'fro')

figure;
hold on;
plot(x, 'b-.');
plot(y, 'r-.');

figure;
plot_bct_struct("H_bct.dat");
