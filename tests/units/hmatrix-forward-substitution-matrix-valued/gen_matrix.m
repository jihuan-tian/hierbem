p = 5;
n = 2^p;
M = randn(n);
## Make the matrix SPD, so that it is always solvable as well as its
## lower triangular part.
M = M * M';
L = tril(M);
save("-text", "L.dat", "L");

Z = rand(n);
save("-text", "Z.dat", "Z");
