clear all;

figure;
subplot(2, 2, 1);
plot_bct_struct("bct1.dat");
title("Block cluster tree #1");

subplot(2, 2, 2);
plot_bct_struct("bct2.dat");
title("Block cluster tree #2");

subplot(2, 2, 3);
plot_bct_struct("bct3.dat");
title("Block cluster tree #3");

subplot(2, 2, 4);
plot_bct_struct("bct4.dat");
title("Block cluster tree #4");

PrintGCF("block-cluster-tensor-product-structure");
