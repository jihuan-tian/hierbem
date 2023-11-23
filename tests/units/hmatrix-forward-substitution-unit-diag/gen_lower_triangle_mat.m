p = 5;
n = 2^p;
M = randn(n);
M = M * M';
## Scale the matrix and make the maximum diagonal entry be 1.
M = M / max(diag(M));
## Get the lower triangular part of the matrix and enforce all diagonal entries
## to be 1.
L = tril(M);
for m = 1:n
  L(m,m) =1.0;
endfor
save("-text", "L.dat", "L");

b = rand(n, 1);
save("-text", "b.dat", "b");
