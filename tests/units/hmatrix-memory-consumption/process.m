clear all;
load "hmatrix-memory-consumption.output";

plot(hmatrix_memory_consumption(:,1), hmatrix_memory_consumption(:,2)/1024/1024, "r-o");
hold on;
plot(hmatrix_memory_consumption(:,1), hmatrix_memory_consumption(:,1).^2*8/1024/1024, "b-+");
grid on;
legend({"Compressed matrix", "Full matrix"}, "location", "southoutside");
xlabel("Matrix size");
ylabel("Memory consumption (MB)");
## title("Linear storage complexity of H-matrix for BEM");
title("Linear storage complexity achieved by compressed matrix\nfor boundary element method");
