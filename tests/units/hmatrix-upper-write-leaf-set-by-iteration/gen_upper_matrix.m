## Generate upper triangular matrix.
M = triu((1:16)' * (1:16));
save("-text", "M.dat", "M");
