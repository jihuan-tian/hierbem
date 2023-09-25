clear all;
load hmatrix-invert-by-gauss-elim.output;

M_inv = inv(M);
H_before_inv_full_inv = inv(H_before_inv_full);

norm(H_before_inv_full_inv - H_inv_full, 'fro') / norm(H_before_inv_full_inv, 'fro')
norm(M_inv - H_inv_full, 'fro') / norm(M_inv, 'fro')

figure;
set_fig_size(gcf, 900, 300);
subplot(1, 3, 1);
plot_bct_struct("H_before_inv_bct.dat");
title("H before inverse");

subplot(1, 3, 2);
plot_bct_struct("H_after_inv_bct.dat");
title("H after inverse");

subplot(1, 3, 3);
plot_bct_struct("H_inv_bct.dat");
title("Inverse of H");

print("hmatrix-invert-by-gauss-elim.png", "-dpng", "-r600");
