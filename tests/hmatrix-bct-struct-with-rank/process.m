clear all;

N = 10;

for m = 1:N
  figure(m);
  plot_bct_struct(cstrcat("bct-struct-with-rank=", num2str(m), ".dat"));
  title(cstrcat("H-matrix block cluster tree structure (rank=", num2str(m), ")"));
  PrintGCF(cstrcat("bct-struc-with-rank=", num2str(m)));
endfor
