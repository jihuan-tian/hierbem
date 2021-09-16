clear all;
load hmatrix-fine-ntp-to-tp.output;

figure;
subplot(3, 2, 1);
plot_bct_struct("bct-fine-ntp.dat");
title("BCT1: fine non-tensor product partition");

subplot(3, 2, 2);
plot_bct_struct("bct-tp.dat");
title("BCT2: tensor product partition");

subplot(3, 2, 3);
plot_bct_struct("hmat1.dat");
title("H-matrix based on BCT1 with rank 2");

subplot(3, 2, 4);
imagesc(M_from_hmat1);
title("H-matrix data");
axis off;

subplot(3, 2, 5);
plot_bct_struct("hmat2.dat");
title("H-matrix converted to BCT2 with rank 1");

subplot(3, 2, 6);
imagesc(M_from_hmat2);
title("H-matrix data after conversion")
axis off;

print("convert_h.png", "-dpng", "-r600");

norm(M - M_from_hmat1, 'fro') / norm(M, 'fro')
norm(M - M_from_hmat2, 'fro') / norm(M, 'fro')
norm(M_from_hmat1 - M_from_hmat2, 'fro') / norm(M_from_hmat1, 'fro')
