p = 5;
n = 2^p;
M = randn(n);
## Make the matrix SPD, so that it is always solvable as well as its
## upper triangular part.
M = M * M';
U = triu(M);
save("-text", "U.dat", "U");

Z = rand(n);
save("-text", "Z.dat", "Z");
