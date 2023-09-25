clear all;
load "M.dat";
load "hmatrix-truncate-to-fixed-rank-preserve-spd.output";

## Restore the matrix M to be symmetric.
MM = tril2fullsym(M);
is_positive_definite(MM);

HH_full_no_spd = tril2fullsym(H_full_no_spd);
is_positive_definite(HH_full_no_spd);

HH_full_before_trunc = tril2fullsym(H_full_before_trunc);
is_positive_definite(HH_full_before_trunc);

HH_full_after_trunc = tril2fullsym(H_full_after_trunc);
is_positive_definite(HH_full_after_trunc);

fprintf(stdout(), "Error between H_full_no_spd and M: ");
norm(HH_full_no_spd - MM, 'fro') / norm(MM, 'fro')
fprintf(stdout(), "Error between H_full_no_spd and H_full_after_trunc: ");
norm(HH_full_no_spd - HH_full_after_trunc, 'fro') / norm(HH_full_after_trunc, 'fro')
fprintf(stdout(), "Error between H_full_before_trunc and M: ");
norm(HH_full_before_trunc - MM, 'fro') / norm(MM, 'fro')
fprintf(stdout(), "Error between H_full_after_trunc and M: ");
norm(HH_full_after_trunc - MM, 'fro') / norm(MM, 'fro')


## figure;
## set_fig_size(gcf, 900, 400);
## subplot(1, 2, 1);
## plot_bct_struct("H_bct_before_trunc.dat");
## axis equal;
## title("H-matrix before the truncation");
## subplot(1, 2, 2);
## plot_bct_struct("H_bct_after_trunc.dat");
## axis equal;
## title("H-matrix after the truncation");

## print("H-matrices.png", "-djpg", "-r800");
