clear all;

A = reshape([1, 3, 5, 7, 9, 10], 3, 2);
B = reshape([2, 8, 9, 7, 1, 3, 11, 20, 13], 3, 3);
C = A' * B
C = reshape([1, 1, 1, 2, 2, 2], 2, 3);
C = C + A' * B
