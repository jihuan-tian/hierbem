clear all;
load U.dat;
load b.dat;
load hmatrix-backward-substitution-unit-diag.output;

y = backward_substitution(U, b);
norm(x - y, 'fro') / norm(y, 'fro')
figure;
hold on;
plot(x, 'b-.');
plot(y, 'r-.');
