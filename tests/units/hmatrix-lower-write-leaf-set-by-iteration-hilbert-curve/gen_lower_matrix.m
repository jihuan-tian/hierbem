## Generate lower triangular matrix.
M = tril((1:16)' * (1:16));
save("-text", "M.dat", "M");
