p = 5;
n = 2^p;
M = randn(n);
L = tril(M);
## Size for the block triangular matrix.
block_size = 2;
for m = 1:2:n
  ## Create a random diagonal block.
  L(m:(m+block_size-1), m:(m+block_size-1)) = randn(block_size);
endfor
save("-text", "L.dat", "L");

b = rand(n, 1);
save("-text", "b.dat", "b");
