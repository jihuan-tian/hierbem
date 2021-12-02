clear all;
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

norm(HH - MM, 'fro') / norm(MM, 'fro')
norm(H_full - M, 'fro') / norm(M, 'fro')
x_octave = MM \ b;
norm(x_octave - x, 2) / norm(x_octave, 2)
x_octave_for_HH = HH \ b;
norm(x_octave_for_HH - x, 2) / norm(x_octave_for_HH, 2)

figure;
plot(x,'ro');
hold on;
plot(x_octave,'b+');
hold off;

figure;
plot(x,'ro');
hold on;
plot(x_octave_for_HH,'b+');
hold off;

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
