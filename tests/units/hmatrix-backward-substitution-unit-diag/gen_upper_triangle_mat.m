p = 5;
n = 2^p;
M = randn(n);
U = triu(M);
for m = 1:n
  U(m,m) = 1.0;
endfor
save("-text", "U.dat", "U");

b = rand(n, 1);
save("-text", "b.dat", "b");
