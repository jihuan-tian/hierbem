clear all;
load L.dat;
load b.dat;
load hmatrix-forward-substitution.output;

y = forward_substitution(L, b);
norm(x - y, 'fro') / norm(y, 'fro')

figure;
hold on;
plot(x, 'b-.');
plot(y, 'r-.');

figure;
plot_bct_struct("H_bct.dat");
