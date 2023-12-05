clear all;

load_packages;
load hmatrix-solve-cholesky-in-situ.output;
load M.dat;
load b.dat;

## Restore the matrix M to be symmetric.
MM = tril2fullsym(M);
is_positive_definite(MM);

## Check the positive definiteness of the full matrix, which is
## converted from the H-matrix in C++ test code.
HH = tril2fullsym(H_full);
is_positive_definite(HH);

printout_var("norm(HH - MM, 'fro') / norm(MM, 'fro')");
printout_var("norm(H_full - M, 'fro') / norm(M, 'fro')");
x_octave = MM \ b;
printout_var("norm(x_octave - x, 2) / norm(x_octave, 2)");
x_octave_for_HH = HH \ b;
printout_var("norm(x_octave_for_HH - x, 2) / norm(x_octave_for_HH, 2)")

figure;
plot(x,'ro');
hold on;
plot(x_octave,'b+');
hold off;
title("Comparison of the solution vector and the real solution");

figure;
plot(x,'ro');
hold on;
plot(x_octave_for_HH,'b+');
hold off;
title("Comparison of the solution vector and the Octave solution vector for the modified matrix");

## figure;
## set_fig_size(gcf, 900, 400);
## subplot(1,2,1);
## plot_bct_struct("H_bct.dat");
## axis equal;
## title("H-matrix");
## subplot(1,2,2);
## plot_bct_struct("LLT_bct.dat");
## axis equal;
## title("Cholesky factorization of H-matrix");

## print("H-matrices.png", "-djpg", "-r800");
