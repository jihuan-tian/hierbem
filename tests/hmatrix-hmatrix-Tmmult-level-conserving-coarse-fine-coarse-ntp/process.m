clear all;
load hmatrix-hmatrix-Tmmult-level-conserving-coarse-fine-coarse-ntp.output;

M = M1' * M2;
norm(H1_mult_H2_full - M, 'fro') / norm(M, 'fro')
norm(H3_full - M, 'fro') / norm(M, 'fro')
