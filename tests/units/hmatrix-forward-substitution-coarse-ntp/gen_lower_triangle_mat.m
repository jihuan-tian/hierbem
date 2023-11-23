p = 5;
M = randn(2^p);
## Make the matrix SPD, so that it is always solvable as well as its
## lower triangular part.
M = M * M';
L = tril(M);
save("-text", "L.dat", "L");

b = rand(2^p, 1);
save("-text", "b.dat", "b");
