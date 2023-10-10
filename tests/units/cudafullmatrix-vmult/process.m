clear all

A = reshape([2, 8, 9, 7, 1, 3, 11, 20, 13], 3, 3);
v = [7, 3, 10]';
w = A * v
w = [1,2,3]';
w = w + A * v
