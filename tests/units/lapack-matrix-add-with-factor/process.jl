A = reshape([1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 3)
B = reshape([3, 5, 7, 4, 6, 8, 5, 7, 9], 3, 3)
b = 3.5

C = A + B * 3.5

A_complex = reshape([1.0 + 0.1im, 2.0 + 0.2im, 3.0 + 0.3im, 4.0 + 0.4im, 5.0 + 0.5im,
                     6.0 + 0.6im, 7.0 + 0.7im, 8.0 + 0.8im, 9.0 + 0.9im], 3, 3)
B_complex = reshape([3.0 + 0.1im, 5.0 + 0.2im, 7.0 + 0.3im, 4.0 + 0.4im, 6.0 + 0.5im,
                     8.0 + 0.6im, 5.0 + 0.7im, 7.0 + 0.8im, 9.0 + 0.9im], 3, 3);
C_complex = A_complex + B_complex * 3.5
