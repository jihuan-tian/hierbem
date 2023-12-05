clear all;

load_packages;
load hmatrix-solve-cholesky.output;
load M.dat;
load b.dat;

## Restore the matrix M to be symmetric.
MM = tril2fullsym(M);

## Check the positive definiteness of the full matrix converted from
## the H-matrix.
HH = tril2fullsym(H_full);
lambda  = eig(HH);
if (min(lambda) < 0)
  fprintf(stdout(), "The matrix is not positive definitive with the minimum eigen value %g!\n", min(lambda));
else
  fprintf(stdout(), "The matrix is positive definitive with the minimum eigen value %g!\n", min(lambda));
endif

norm(LLT_full*LLT_full' - MM, 'fro') / norm(MM, 'fro')
norm(H_full - M, 'fro') / norm(M, 'fro')
x_octave = MM \ b;
norm(x_octave - x, 2) / norm(x_octave, 2)

figure;
plot(x,'ro');
hold on;
plot(x_octave,'b+');
hold off;

figure;
set_fig_size(gcf, 900, 400);
subplot(1,2,1);
plot_bct_struct("H_bct.dat");
title("H-matrix");
subplot(1,2,2);
plot_bct_struct("LLT_bct.dat");
title("Cholesky factorization of H-matrix");

figure;
show_matrix(LLT_full);
hold on;
plot_bct_struct("LLT_bct.dat", false, false);
