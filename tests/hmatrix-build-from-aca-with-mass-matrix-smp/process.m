clear all;
## Load SLP and DLP full matrix data that are directly evaluated from
## the Galerkin-BEM double integral.
load matrices-assemble-on-cell-pair-with-mass-matrix.dat;
## Load SLP and DLP full mtrix data that are converted from
## corresponding H-matrices.
load hmatrix-build-from-aca-with-mass-matrix-smp_fine-mesh.output.dat;

figure;
subplot(1, 2, 1);
show_matrix(dlp_cell_pair);
colorbar;
title("DLP direct evaluation");
subplot(1, 2, 2);
show_matrix(dlp_full);
colorbar;
title("DLP from H-matrix");
PrintGCF("dlp-matrix");

figure;
subplot(1, 2, 1);
show_matrix(slp_cell_pair);
colorbar;
title("SLP direct evaluation");
subplot(1, 2, 2);
show_matrix(slp_full);
colorbar;
title("SLP from H-matrix");
PrintGCF("slp-matrix");

norm(dlp_full - dlp_cell_pair, 'fro') / norm(dlp_cell_pair, 'fro')
norm(diag(dlp_full) - diag(dlp_cell_pair)) / norm(diag(dlp_cell_pair))
norm(slp_full - slp_cell_pair, 'fro') / norm(slp_cell_pair, 'fro')
norm(diag(slp_full) - diag(slp_cell_pair)) / norm(diag(slp_cell_pair))

figure;
bar([diag(dlp_full)(1:10), diag(dlp_cell_pair)(1:10)])
legend({"DLP from H-matrix", "DLP direct evaluation"});

figure;
bar([diag(slp_full)(1:10), diag(slp_cell_pair)(1:10)])
legend({"SLP from H-matrix", "SLP direct evaluation"});
