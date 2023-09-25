A = [-1, 5,  2,  -3, 6,  1,  -2, 4,  2, -3, -4, 1,  -3, -1, 1,  2,  -2, 4,  2,  -1, 3,  1,  -1, 3,  -3, 7, 2,  -3, 7,  2,  -2, 2,  1,  0,  0,  -1, 1,  -4, 0, 0,  0,  2,  0,  -2, 3,  -1, -1, 6,  -2, 4,  3,  -2, 4,  -1, -1, 3,  3,  -4, -6, 1,  -3, -3, 1,  -2];
A = reshape(A, 8, 8);
A_inv = inv(A);

A_inv_cpp = load('cudafullmatrix-invert-by-gauss-elim.output');
printout_var('norm(A_inv - A_inv_cpp, 2) / norm(A_inv, 2)');
