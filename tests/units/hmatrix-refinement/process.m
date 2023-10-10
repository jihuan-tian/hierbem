clear all;
load hmatrix-refinement.output;

figure;
subplot(2, 2, 1);
plot_bct_struct("bct-coarse.dat");
title("Coarse partition");
subplot(2, 2, 2);
plot_bct_struct("bct-fine.dat");
title("Fine partition");
subplot(2, 2, 3);
plot_bct_struct("hmat-coarse.dat");
title("Matrix built on coarse partition");
subplot(2, 2, 4);
plot_bct_struct("hmat-fine.dat");
title("Matrix extended to fine partition");

PrintGCF("partition-structure");

## There is no accuracy loss during the refinement of an H-matrix.
## Hence, the full matrices converted from the two H-matrices before
## and after the refinement operation should be exactly the same.
norm(M_from_hmat_coarse - M_from_hmat_fine, 'fro') / norm(M_from_hmat_coarse, 'fro')

## Compare the converted full matrix from the H-matrix with the
## original full matrix.
norm(M_from_hmat_fine - M, 'fro') / norm(M, 'fro')
