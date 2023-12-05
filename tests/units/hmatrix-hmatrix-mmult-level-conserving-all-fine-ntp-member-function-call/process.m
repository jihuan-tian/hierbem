clear all;

load_packages;
load hmatrix-hmatrix-mmult-level-conserving-all-fine-ntp-member-function-call.output;

M = M1 * M2;
norm(H1_mult_H2_full - M, 'fro') / norm(M, 'fro')
norm(H3_full - M, 'fro') / norm(M, 'fro')

figure;
set_fig_size(gcf, 900, 300);
subplot(1, 3, 1);
plot_bct_struct("H1_bct.dat");
title("H1");
subplot(1, 3, 2);
plot_bct_struct("H2_bct.dat");
title("H2");
subplot(1, 3, 3);
plot_bct_struct("H3_bct.dat");
title("H3=H1*H2");

print("hmatrix-hmatrix-mmult-level-conserving-all-fine-ntp-member-function-call.png", "-dpng", "-r600");
