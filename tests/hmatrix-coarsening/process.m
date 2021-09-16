clear all;
load hmatrix-coarsening.output;

norm(hmat_fine_to_full - M, 'fro') / norm(M, 'fro')
norm(hmat_coarse_to_full - M, 'fro') / norm(M, 'fro')

figure;
plot_bct_struct("bct1.dat", false);
title("Fine block cluster tree structure");
figure;
plot_bct_struct("bct2.dat", false);
title("Coarse block cluster tree structure");
figure;
plot_bct_struct("hmat_fine_partition.dat", true);
title("Fine H-matrix structure");
figure;
plot_bct_struct("hmat_coarse_partition.dat", true);
title("Coarse H-matrix structure");
