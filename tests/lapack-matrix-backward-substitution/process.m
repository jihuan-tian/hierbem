U = [1,0,0,0;2,3,0,0;4,5,6,0;7,8,9,10]';
b = [3;6;9;10];
y = backward_substitution(U, b)
norm(U * y - b, 'fro') / norm(b, 'fro')
