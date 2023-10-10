L = [1,0,0,0;2,1,0,0;4,5,1,0;7,8,9,1];
b = [3;6;9;10];
y = forward_substitution(L, b)
norm(L * y - b, 'fro') / norm(b, 'fro')
