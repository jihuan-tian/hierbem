p = 5;
n = 2^p;
M = randn(n);
L = tril(M);
for m = 1:n
  L(m,m) =1.0;
endfor
save("-text", "L.dat", "L");

b = rand(n, 1);
save("-text", "b.dat", "b");
