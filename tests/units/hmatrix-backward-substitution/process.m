clear all;
load U.dat;
load b.dat;
load hmatrix-backward-substitution.output;

y = backward_substitution(U, b);
norm(x - y, 'fro') / norm(y, 'fro')
