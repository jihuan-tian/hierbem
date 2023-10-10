clear all;
load U.dat;
load b.dat;
load hmatrix-backward-substitution-coarse-ntp.output;

y = backward_substitution(U, b);
norm(x - y, 'fro') / norm(y, 'fro')

figure;
hold on;
plot(x, 'b-.');
plot(y, 'r-.');

figure;
plot_bct_struct("H_bct.dat");
