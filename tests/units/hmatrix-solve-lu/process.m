clear all;
load hmatrix-solve-lu.output;
load M.dat;
load b.dat;

norm(H_full - M, 'fro') / norm(M, 'fro')
x_octave = M \ b;
norm(x_octave - x, 2) / norm(x_octave, 2)

figure;
plot(x,'r-', "marker", "+");
hold on;
plot(x_octave,'g-', "marker", "^");
hold off;

figure;
set_fig_size(gcf, 900, 400);
subplot(1,2,1);
plot_bct_struct("H_bct.dat");
title("H-matrix");
subplot(1,2,2);
plot_bct_struct("LU_bct.dat");
title("LU factorization of H-matrix");

print("H-matrices.png", "-djpg", "-r800");
