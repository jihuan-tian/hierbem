clear all;
load L.dat;
load Z.dat;
load hmatrix-forward-substitution-matrix-valued.output;

X_octave = L \ Z;
norm(X - X_octave, 'fro') / norm(X_octave, 'fro')

figure;
set_fig_size(gcf, 900, 300);
subplot(1, 3, 1);
plot_bct_struct("HL_bct.dat");
title("Lower triangular matrix");
subplot(1, 3, 2);
plot_bct_struct("HZ_bct.dat");
title("RHS matrix");
subplot(1, 3, 3);
plot_bct_struct("HX_bct.dat");
title("Solution matrix");

print(gcf, "matrix_bct.png", "-djpg", "-r600");
